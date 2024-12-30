pub fn sgd(
    mut w: Vec<f64>,
    dw: &[f64],
    config: Option<&mut HashMap<String, f64>>,
) -> (Vec<f64>, HashMap<String, f64>) {
    let mut config = config.unwrap_or(&mut HashMap::new()).clone();
    let learning_rate = config.entry("learning_rate".to_string()).or_insert(1e-2);

    for (wi, &dwi) in w.iter_mut().zip(dw.iter()) {
        *wi -= *learning_rate * dwi;
    }

    (w, config)
}