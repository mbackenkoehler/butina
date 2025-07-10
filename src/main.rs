use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, ArrayView1};
use ndarray_npy::read_npy;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::sync::Mutex;

fn tanimoto(fp1: ArrayView1<u8>, fp2: ArrayView1<u8>) -> f64 {
    let intersect = fp1
        .iter()
        .zip(fp2.iter())
        .map(|(a, b)| (a & b).count_ones())
        .sum::<u32>();
    let union = fp1.iter().map(|x| x.count_ones()).sum::<u32>()
        + fp2.iter().map(|x| x.count_ones()).sum::<u32>()
        - intersect;
    if union == 0 {
        0.0
    } else {
        intersect as f64 / union as f64
    }
}

fn butina(fps: &Array2<u8>, threshold: f64) -> Vec<Vec<usize>> {
    let n = fps.nrows();
    let mut neighbors = vec![vec![]; n];

    let total_pairs = n * (n - 1) / 2;
    let bar = ProgressBar::new(total_pairs as u64);
    bar.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] [{wide_bar}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    let sim_results = Mutex::new(vec![]);

    (0..n).into_par_iter().for_each(|i| {
        let row_i = fps.row(i);
        let mut local_pairs = Vec::new();

        for j in (i + 1)..n {
            let sim = tanimoto(row_i, fps.row(j));
            if sim >= threshold {
                local_pairs.push((i, j, sim));
            }
            bar.inc(1);
        }

        let mut guard = sim_results.lock().unwrap();
        guard.extend(local_pairs);
    });

    bar.finish_with_message("Similarity matrix computed");

    for (i, j, sim) in sim_results.into_inner().unwrap() {
        neighbors[i].push((j, sim));
        neighbors[j].push((i, sim));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| -(neighbors[i].len() as isize));

    let mut used = HashSet::new();
    let mut clusters = Vec::new();

    for &i in &order {
        if used.contains(&i) {
            continue;
        }
        let mut cluster = vec![i];
        used.insert(i);
        for &(j, sim) in &neighbors[i] {
            if !used.contains(&j) && sim >= threshold {
                cluster.push(j);
                used.insert(j);
            }
        }
        clusters.push(cluster);
    }

    clusters
}

fn main() -> Result<()> {
    let fps: Array2<u8> = read_npy("fingerprints.npy")?;
    let smiles: Vec<String> = BufReader::new(File::open("smiles.txt")?)
        .lines()
        .filter_map(Result::ok)
        .collect();

    assert_eq!(smiles.len(), fps.nrows());

    println!(
        "Running Butina clustering on {} fingerprints...",
        fps.nrows()
    );
    let clusters = butina(&fps, 0.7);
    println!("Clustering complete.");

    let mut smiles_with_cluster = vec![("".to_string(), -1isize); smiles.len()];
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &idx in cluster {
            smiles_with_cluster[idx] = (smiles[idx].clone(), cluster_id as isize);
        }
    }

    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("clusters.csv")?;
    writeln!(out, "SMILES,cluster")?;
    for (smi, clust) in &smiles_with_cluster {
        writeln!(out, "{},{}", smi, clust)?;
    }

    println!("✅ Wrote output to clusters.csv");

    // Cluster stats
    let mut size_dist: HashMap<usize, usize> = HashMap::new();
    for cluster in &clusters {
        *size_dist.entry(cluster.len()).or_default() += 1;
    }

    println!("\n📊 Cluster statistics:");
    println!("  Total clusters: {}", clusters.len());
    println!("  Singleton clusters: {}", size_dist.get(&1).unwrap_or(&0));
    println!("  Cluster size distribution:");
    let mut sizes: Vec<_> = size_dist.into_iter().collect();
    sizes.sort_by_key(|(size, _)| *size);
    for (size, count) in sizes {
        println!("    size {:>2}: {:>4} clusters", size, count);
    }

    Ok(())
}
