pub fn sgd_momentum(
    mut w: Vec<f64>,
    dw: &[f64],
    config: Option<&mut HashMap<String, Vec<f64>>>,
) -> (Vec<f64>, HashMap<String, Vec<f64>>) {
    let mut config = config.unwrap_or(&mut HashMap::new()).clone();
    let learning_rate = *config.entry("learning_rate".to_string()).or_insert(vec![1e-2])[0];
    let momentum = *config.entry("momentum".to_string()).or_insert(vec![0.9])[0];
    let velocity = config.entry("velocity".to_string()).or_insert(vec![0.0; w.len()]);

    for (i, (&dwi, vi)) in dw.iter().zip(velocity.iter_mut()).enumerate() {
        *vi = momentum * *vi - learning_rate * dwi;
        w[i] += *vi;
    }

    config.insert("velocity".to_string(), velocity.clone());
    (w, config)
}