pub fn relu_forward(x: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    // Computes the forward pass for a layer of rectified linear units (ReLUs).
    let mut out = x.clone();
    for value in out.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    let cache = x.clone();
    (out, cache)
}

pub fn relu_backward(dout: Vec<f64>, cache: Vec<f64>) -> Vec<f64> {
    // Computes the backward pass for a layer of rectified linear units (ReLUs).
    let mut dx = dout.clone();
    for (i, &value) in cache.iter().enumerate() {
        dx[i] = if value > 0.0 { dx[i] } else { 0.0 };
    }
    dx
}
