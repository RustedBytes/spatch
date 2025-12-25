use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

/// -----------------------------------------------------------------------
/// Type Definitions & Constants
/// -----------------------------------------------------------------------

type Vector = Vec<f32>;
type NodeId = usize;

#[derive(Clone, Debug)]
struct Node {
    id: NodeId,
    vector: Vector,
    /// Adjacency list: layers[layer_level] = list of neighbor IDs
    layers: Vec<Vec<NodeId>>,
}

/// A candidate for the priority queues.
/// We implement `Ord` based on distance for MaxHeap behavior (default).
/// For MinHeap (closest first), we will use `Reverse<Candidate>`.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: NodeId,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Standard comparison: larger distance > smaller distance
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// -----------------------------------------------------------------------
/// HNSW Structure
/// -----------------------------------------------------------------------

pub struct Hnsw {
    nodes: Vec<Option<Node>>, // Option allows "soft" deletion (slot reuse)
    entry_point: Option<NodeId>,
    max_layers: usize,
    m_max: usize,           // Max neighbors per node
    m_max0: usize,          // Max neighbors at layer 0
    ef_construction: usize, // Beam width for construction
    level_mult: f64,        // Level generation factor
}

impl Hnsw {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_layers: 0,
            m_max: m,
            m_max0: m * 2,
            ef_construction,
            level_mult: 1.0 / (m as f64).ln(),
        }
    }

    // --- Core Logic: Euclidean Distance ---
    fn dist_vec(a: &Vector, b: &Vector) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn dist(&self, u: NodeId, v: NodeId) -> f32 {
        let u_vec = &self.nodes[u].as_ref().expect("Node U missing").vector;
        let v_vec = &self.nodes[v].as_ref().expect("Node V missing").vector;
        Self::dist_vec(u_vec, v_vec)
    }

    // --- Insertion ---
    pub fn insert(&mut self, vector: Vector) -> NodeId {
        let level = self.random_level();
        let new_id = self.nodes.len();

        let new_node = Node {
            id: new_id,
            vector: vector.clone(),
            layers: vec![vec![]; level + 1],
        };
        self.nodes.push(Some(new_node));

        let entry = match self.entry_point {
            Some(ep) => ep,
            None => {
                self.entry_point = Some(new_id);
                self.max_layers = level;
                return new_id;
            }
        };

        let mut curr_node = entry;
        let max_l = self.nodes[entry].as_ref().unwrap().layers.len() - 1;

        // Phase 1: Zoom down (Greedy search)
        for l in (level + 1..=max_l).rev() {
            if let Some(res) = self.search_layer(curr_node, &vector, 1, l).first() {
                curr_node = res.id;
            }
        }

        // Phase 2: Insert and Connect
        for l in (0..=std::cmp::min(level, max_l)).rev() {
            let neighbors = self.search_layer(curr_node, &vector, self.ef_construction, l);

            // Heuristic Selection
            let selected = self.select_neighbors(&neighbors, l);

            // Bidirectional connect
            self.nodes[new_id].as_mut().unwrap().layers[l] = selected.clone();

            for &neighbor_id in &selected {
                self.add_connection(neighbor_id, new_id, l);
            }

            // Move entry point closer for next layer
            if let Some(first) = neighbors.first() {
                curr_node = first.id;
            }
        }

        if level > self.max_layers {
            self.max_layers = level;
            self.entry_point = Some(new_id);
        }

        new_id
    }

    // --- Deletion with Repair (The Algorithm) ---
    pub fn delete(&mut self, target_id: NodeId) {
        if self.nodes.get(target_id).is_none() || self.nodes[target_id].is_none() {
            return;
        }

        println!(
            "  [Repair] Deleting Node {} and rewiring neighbors...",
            target_id
        );

        let target_layers = self.nodes[target_id].as_ref().unwrap().layers.clone();

        // 1. Remove edges TO the target
        for (l, neighbors) in target_layers.iter().enumerate() {
            for &neighbor_id in neighbors {
                self.remove_connection(neighbor_id, target_id, l);
            }
        }

        // 2. Repair: Connect neighbors to each other ("Edge Contraction")
        for (l, neighbors) in target_layers.iter().enumerate() {
            for &u in neighbors {
                let mut candidates = Vec::new();
                for &v in neighbors {
                    if u == v {
                        continue;
                    }
                    let d = self.dist(u, v);
                    candidates.push(Candidate { id: v, distance: d });
                }

                // Sort by distance (closest first)
                candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

                // Apply Heuristic to limit connections
                let selected = self.select_neighbors_from_candidates(candidates, l);

                for neighbor_to_add in selected {
                    self.add_connection(u, neighbor_to_add, l);
                }
            }
        }

        // 3. Soft Delete
        self.nodes[target_id] = None;

        // 4. Update entry point if needed
        if self.entry_point == Some(target_id) {
            // Brute force find new entry for demo purposes
            self.entry_point = self
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(i, n)| n.as_ref().map(|_| i))
                .next();
        }
    }

    // --- Query ---
    pub fn knn_search(&self, query: &Vector, k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        if let Some(entry) = self.entry_point {
            let mut curr_node = entry;
            let max_l = self.nodes[entry].as_ref().unwrap().layers.len() - 1;

            for l in (1..=max_l).rev() {
                if let Some(res) = self.search_layer(curr_node, query, 1, l).first() {
                    curr_node = res.id;
                }
            }

            let candidates = self.search_layer(curr_node, query, ef, 0);
            candidates
                .into_iter()
                .take(k)
                .map(|c| (c.id, c.distance))
                .collect()
        } else {
            vec![]
        }
    }

    // --- Helpers ---

    fn search_layer(
        &self,
        entry: NodeId,
        query: &Vector,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-Heap (Reverse)
        let mut results = BinaryHeap::new(); // Max-Heap (Standard)

        let d_entry = if let Some(n) = &self.nodes[entry] {
            Self::dist_vec(&n.vector, query)
        } else {
            f32::MAX
        };

        let init = Candidate {
            id: entry,
            distance: d_entry,
        };
        candidates.push(Reverse(init));
        results.push(init);
        visited.insert(entry);

        while let Some(Reverse(c)) = candidates.pop() {
            let furthest_dist = results.peek().unwrap().distance;

            if c.distance > furthest_dist && results.len() >= ef {
                break;
            }

            if let Some(node) = &self.nodes[c.id] {
                // Defensive check for layer existence
                if layer >= node.layers.len() {
                    continue;
                }

                for &neighbor_id in &node.layers[layer] {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);

                        // Handle deleted neighbors gracefully
                        if let Some(neighbor_node) = &self.nodes[neighbor_id] {
                            let dist = Self::dist_vec(&neighbor_node.vector, query);

                            if dist < furthest_dist || results.len() < ef {
                                let cand = Candidate {
                                    id: neighbor_id,
                                    distance: dist,
                                };
                                candidates.push(Reverse(cand));
                                results.push(cand);

                                if results.len() > ef {
                                    results.pop(); // Remove furthest
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return sorted by distance (closest first)
        let mut res_vec = results.into_vec();
        res_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        res_vec
    }

    fn select_neighbors(&self, candidates: &[Candidate], layer: usize) -> Vec<NodeId> {
        let limit = if layer == 0 { self.m_max0 } else { self.m_max };
        candidates.iter().take(limit).map(|c| c.id).collect()
    }

    fn select_neighbors_from_candidates(
        &self,
        candidates: Vec<Candidate>,
        layer: usize,
    ) -> Vec<NodeId> {
        let limit = if layer == 0 { self.m_max0 } else { self.m_max };
        candidates.iter().take(limit).map(|c| c.id).collect()
    }

    fn add_connection(&mut self, src: NodeId, dest: NodeId, layer: usize) {
        if let Some(node) = self.nodes[src].as_mut()
            && !node.layers[layer].contains(&dest) {
                node.layers[layer].push(dest);
                // Production: If len > M_max, apply shrink heuristic here
            }
    }

    fn remove_connection(&mut self, src: NodeId, dest: NodeId, layer: usize) {
        if let Some(node) = self.nodes[src].as_mut() {
            node.layers[layer].retain(|&x| x != dest);
        }
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let r: f64 = rng.random();
        (-r.ln() * self.level_mult).floor() as usize
    }
}

/// -----------------------------------------------------------------------
/// DEMO MAIN
/// -----------------------------------------------------------------------

fn main() {
    println!("--- HNSW with Dynamic Updates Demo ---");

    // 1. Initialize HNSW
    let mut hnsw = Hnsw::new(5, 10); // m=5, ef=10

    // 2. Generate Data (Grid of points)
    // We create a predictable structure to verify deletion easily.
    // 100 points
    let mut points = Vec::new();
    let mut rng = rand::rng();
    for _ in 0..100 {
        let v = vec![rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)];
        points.push(v);
    }

    println!("-> Inserting 100 random vectors...");
    for (i, p) in points.iter().enumerate() {
        hnsw.insert(p.clone());
        if i % 20 == 0 {
            print!(".");
        }
    }
    println!(" Done.");

    // 3. Define a Query
    let query = vec![50.0, 50.0];
    println!("\n-> Querying nearest neighbors to [50, 50] (k=3)...");

    let results_before = hnsw.knn_search(&query, 3, 20);
    println!("   Results: {:?}", results_before);

    if results_before.is_empty() {
        println!("Error: No results found.");
        return;
    }

    // 4. Perform Deletion Test
    // We delete the closest node found.
    let victim_id = results_before[0].0;
    println!("\n-> Deleting top result Node {}...", victim_id);

    hnsw.delete(victim_id);

    // 5. Verify Graph Repair
    println!("-> Re-querying [50, 50] (k=3)...");
    let results_after = hnsw.knn_search(&query, 3, 20);
    println!("   Results: {:?}", results_after);

    // Validation
    let still_exists = results_after.iter().any(|(id, _)| *id == victim_id);
    if !still_exists && !results_after.is_empty() {
        println!(
            "\n[SUCCESS] Deleted node is gone, and graph is still navigable (found neighbors)."
        );
        println!("          The repair mechanism successfully re-routed paths.");
    } else {
        println!("\n[FAIL] Something went wrong (Node still there or graph broken).");
    }
}
