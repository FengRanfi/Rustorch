
pub fn rmsprop(
    mut w: Vec<f64>,
    dw: &[f64],
    config: Option<&mut HashMap<String, f64>>,
) -> (Vec<f64>, HashMap<String, f64>) {
    let mut config = config.unwrap_or(&mut HashMap::new()).clone();
    let learning_rate = config.entry("learning_rate".to_string()).or_insert(1e-2);
    let decay_rate = config.entry("decay_rate".to_string()).or_insert(0.99);
    let epsilon = config.entry("epsilon".to_string()).or_insert(1e-8);
    let cache = config.entry("cache".to_string()).or_insert_with(|| vec![0.0; w.len()]);

    for (i, (&dwi, cachei)) in dw.iter().zip(cache.iter_mut()).enumerate() {
        *cachei = decay_rate * *cachei + (1.0 - decay_rate) * dwi * dwi;
        w[i] -= learning_rate / ((*cachei + epsilon).sqrt()) * dwi;
    }

    config.insert("cache".to_string(), cache.clone());
    (w, config)
}