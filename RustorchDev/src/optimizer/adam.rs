pub fn adam(
    mut w: Vec<f64>,
    dw: &[f64],
    config: Option<&mut HashMap<String, f64>>,
) -> (Vec<f64>, HashMap<String, f64>) {
    let mut config = config.unwrap_or(&mut HashMap::new()).clone();
    let learning_rate = config.entry("learning_rate".to_string()).or_insert(1e-3);
    let beta1 = config.entry("beta1".to_string()).or_insert(0.9);
    let beta2 = config.entry("beta2".to_string()).or_insert(0.999);
    let epsilon = config.entry("epsilon".to_string()).or_insert(1e-8);
    let m = config.entry("m".to_string()).or_insert_with(|| vec![0.0; w.len()]);
    let v = config.entry("v".to_string()).or_insert_with(|| vec![0.0; w.len()]);
    let t = config.entry("t".to_string()).or_insert(0.0);

    *t += 1.0;
    let t_corr = *t;

    for (i, (&dwi, (mi, vi))) in dw.iter().zip(m.iter_mut().zip(v.iter_mut())).enumerate() {
        *mi = beta1 * *mi + (1.0 - beta1) * dwi;
        *vi = beta2 * *vi + (1.0 - beta2) * dwi * dwi;
        let m_hat = *mi / (1.0 - beta1.powf(t_corr));
        let v_hat = *vi / (1.0 - beta2.powf(t_corr));
        w[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
    }

    config.insert("m".to_string(), m.clone());
    config.insert("v".to_string(), v.clone());
    config.insert("t".to_string(), t_corr);

    (w, config)
}
