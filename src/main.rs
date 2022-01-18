//use numpy::PyArray1;
use pyo3::prelude::{PyResult, Python};
use pyo3::types::{PyDict, IntoPyDict};
use std::collections::HashMap;
//use pyo3::conversion::{ToPyObject, IntoPy};
//use ndarray::prelude::*;

fn uturn(q_plus: &Vec<f64>, q_minus: &Vec<f64>, p_plus: &Vec<f64>, p_minus: &Vec<f64>) -> bool {
    let uturn_forward = itertools::izip!(p_plus, q_plus, q_minus).map(|(p_plus_e, q_plus_e, q_minus_e)| p_plus_e * (q_plus_e - q_minus_e)).sum::<f64>() <= 0.0;
    let uturn_backward = itertools::izip!(p_minus, q_plus, q_minus).map(|(p_minus_e, q_plus_e, q_minus_e)| -p_minus_e * (q_minus_e - q_plus_e)).sum::<f64>() <= 0.0;

    uturn_forward && uturn_backward
}

fn kinetic_energy(p : &Vec<f64>, diag_m_inv : &Vec<f64>) -> f64 {
    let sum : f64 = p.iter().zip(diag_m_inv.iter()).map(|(x, m_inv)| x * x * m_inv).sum();
    0.5 * sum
}

fn value_and_grad(q : &Vec<f64>) -> (f64, Vec<f64>) {
    let sum : f64 = q.iter().map(|x| x * x).sum();
    
    (0.5 * sum, q.clone())
}

fn log_sum_exp(x : &Vec<f64>) -> f64 {
    let y = x.iter().fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));
    y + x.iter().map(|xv| (xv - y).exp()).sum::<f64>().ln()
}
  
fn softmax(x : &Vec<f64>) -> Vec<f64>{
    Vec::from_iter(x.iter().map(|xv| (xv - log_sum_exp(x)).exp()))
}

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let locals = PyDict::new(py);
        py.run(
            r#"
import numpy
rng = numpy.random.default_rng(seed = 1)
current_draw = numpy.zeros(2)
stepsize = 0.05
diagonal_inverse_metric = numpy.array([0.3, 1.7])
max_treedepth = 10
"#,
            None,
            Some(locals),
        )
        .unwrap();

        let py_rng = locals.get_item("rng").unwrap();

        let max_treedepth : usize = locals.get_item("max_treedepth").unwrap().extract()?;
        let stepsize : f64 = locals.get_item("stepsize").unwrap().extract()?;
        let q0 : Vec<f64> = locals.get_item("current_draw").unwrap().extract()?;
        let diag_m_inv : Vec<f64> = locals.get_item("diagonal_inverse_metric").unwrap().extract()?;

        let size = q0.len();
        let z : Vec<f64> = py_rng.call_method1("normal", (0.0, 1.0, size)).unwrap().extract()?;
        let directions : Vec<i32> = py_rng.call_method1("integers", (0, 2, max_treedepth + 1)).unwrap().extract()?;
        let kwargs = [("size", max_treedepth)].into_py_dict(py);
        let choice_draws : Vec<f64> = py_rng.call_method("random", (), Some(kwargs)).unwrap().extract()?;
        let pick_draws : Vec<f64> = py_rng.call_method("random", (), Some(kwargs)).unwrap().extract()?;

        //let q0 = Array1::from_iter(q0_vec);
        //let diag_m_inv = Array1::from_iter(diag_m_inv_vec);
        //let z = Array1::from_iter(z_vec);
        let p0 = Vec::from_iter(z.iter().zip(diag_m_inv.iter()).map(|(z, m_inv)| z / m_inv.sqrt()));

        let (u0, grad0) = value_and_grad(&q0);
        let h0 = kinetic_energy(&p0, &diag_m_inv) + u0;

        // directions is a length max_treedepth vector that maps each treedepth
        //  to an integration direction (left or right)
        let directions : Vec<i32> = Vec::from_iter(directions.iter().map(|d| if *d == 0 {-1} else {1}));
        
        // depth_map will be a vector of length 2^max_treedepth that maps each of
        //  the possibly 2^max_treedepth points to a treedepth
        let mut depth_map = vec![0];
        for depth in 1..(max_treedepth + 1) {
            let direction = directions[depth];

            let mut new_section = vec![depth; depth_map.len()];

            if direction < 0 {
                new_section.append(&mut depth_map);
                depth_map = new_section;
            } else {
                depth_map.append(&mut new_section);
            }
        }

        // Steps is a dict that maps treedepth to which leapfrog steps were
        //  computed in that treedepth (kinda the opposite of depth_map)
        let mut steps : HashMap<usize, Vec<usize>> = HashMap::new();

        // Apparently using as for the conversion is kinda bad cause it can mess up
        let max_leapfrogs : usize = 2_usize.pow(max_treedepth as u32);

        // qs stores our positions
        let mut qs = vec![vec![0_f64; size]; max_leapfrogs];
        // ps stores our momentums
        let mut ps = vec![vec![0_f64; size]; max_leapfrogs];

        // log_pi defined in section A.2.3 of https://arxiv.org/abs/1701.02434
        let mut log_pi = vec![0_f64; max_leapfrogs];
        // index of initial state
        let (i_first, _) = depth_map.iter().enumerate().find(|(pos, &d)| d == 0).unwrap();

        qs[i_first] = q0;
        ps[i_first] = p0;
        log_pi[i_first] = -h0;
        
        let mut divergence = false;
        let mut accept_stat : Option<f64> = None;
        // i_left and i_right are indices that track the leftmost and rightmost
        //  states of the integrated trajectory
        let mut i_left = i_first;
        let mut i_right = i_first;
        // log_sum_pi_old is the log of the sum of the pis (of log_pi) for the
        //  tree processed so far
        let mut log_sum_pi_old = log_pi[i_first];
        // i_old will be the sample chosen (the sample from T(z' | told) in section
        //  A.3.1 of https://arxiv.org/abs/1701.02434)
        let mut i_old = i_first;
        // We need to know whether we terminated the trajectory cause of a uturn or we
        //  hit the max trajectory length
        let mut uturn_detected = false;
        // For trees of increasing treedepth

        for depth in 1..(max_treedepth + 1) {
            // Figure out what leapfrog steps we need to compute. If integrating in the
            //  positive direction update the index that points at the right side of the
            //  trajectory. If integrating in the negative direction update the index pointing
            //  to the left side of the trajectory.
            let depth_steps = if directions[depth] < 0 {
                Vec::from_iter(depth_map.iter().enumerate().filter_map(|(pos, &d)| if d == depth {Some(pos)} else {None}).rev())
            } else {
                Vec::from_iter(depth_map.iter().enumerate().filter_map(|(pos, &d)| if d == depth {Some(pos)} else {None}))
            };
            
            if directions[depth] < 0 {
                i_left = *depth_steps.last().unwrap();
            } else {
                i_right = *depth_steps.last().unwrap();
            }

            // Unused except for debugging so I'm getting rid of it
            // See: https://stackoverflow.com/a/30414450
            //steps[&depth] = depth_steps

            let mut checks : Vec<(usize, usize)> = Vec::new();
            // What we're doing here is generating a trajectory composed of a number of leapfrog states.
            // We apply a set of comparisons on this trajectory that can be organized to look like a binary tree.
            // Sometimes I say trajectory instead of tree. Each piece of the tree in the comparison corresponds
            //  to a subset of the trajectory. When I say trajectory I'm referring to a piece of the trajectory that
            //  also corresponds to some sorta subtree in the comparisons.
            //
            // This is probably confusing but what I want to communicate is trajectory and tree are very related
            //  but maybe technically not the same thing.
            //
            // Detect U-turns in newly integrated subtree
            let mut uturn_detected_new_tree = false;

            // Starts and ends are relative because they point to the ith leapfrog step in a sub-trajectory
            //  of size 2^tree_depth which needs to be mapped to the global index of qs
            //  (which is defined in the range 1:2^max_treedepth)
            //
            // We're assuming at this point that depth_steps is in sorted order
            //
            // The sort is necessary because depth_steps is sorted in order of leapfrog steps taken
            //  which might be backwards in time (so decreasing instead of increasing)
            //
            // This sort is important because we need to keep track of what is left and what is right
            //  in the trajectory so that we do the right comparisons
            if depth > 1 {
                // Start at root of new subtree and work down to leaves
                for uturn_depth in (1..depth).rev() {
                    // The root of the comparison tree compares the leftmost to the rightmost states of the new
                    //  part of the trajectory.
                    //  The next level down in the tree of comparisons cuts that trajectory in two and compares
                    //  the leftmost and rightmost elements of those smaller trajectories.
                    //  Etc. Etc.
                    let div_length = 2_u32.pow(uturn_depth as u32) as usize;

                    // Starts are relative indices pointing to the leftmost state of each comparison to be done
                    for start in (0..depth_steps.len()).step_by(div_length) {
                        // Ends are relative indices pointing to the rightmost state for each comparison to be done
                        let end = start + div_length - 1;
                        checks.push((start, end));
                    }
                }
            }

            // Sort the checks so that we only need to check them in order
            checks.sort_by_key(|&(_, end)| end);

            let dt = stepsize * directions[depth] as f64;

            let i_prev = (depth_steps[0] as i32 - directions[depth]) as usize;

            let mut q = qs[i_prev].clone();
            let mut p = ps[i_prev].clone();

            let (_, grad) = value_and_grad(&q);

            // Initialize pointer into checks list
            let mut check_i = 0;

            // These are a bunch of temporaries to minimize numpy
            // allocations during integration
            //let mut p_half = vec![0; size];
            let half_dt = dt / 2.0;
            let mut half_dt_grad = Vec::from_iter(grad.iter().map(|x| x * half_dt));
            let dt_diag_m_inv = Vec::from_iter(diag_m_inv.iter().map(|x| x * dt));
            let mut leapfrogs_taken = 0;
            for i in depth_steps.iter() {
                // leapfrog step
                leapfrogs_taken += 1;

                // p_half = p - (dt / 2) * grad
                for (pv, h) in itertools::izip!(&mut p, &half_dt_grad) {
                    *pv -= h
                }   // p here is actually p_half
                // q = q + dt * numpy.dot(M_inv, p_half)
                // q = q + dt * diag_M_inv * p_half
                let zz = 1;
                for (qv, h, pv) in itertools::izip!(&mut q, &dt_diag_m_inv, &p) {
                    *qv += h * pv;
                }

                let (u, grad) = value_and_grad(&q);
                half_dt_grad = Vec::from_iter(grad.iter().map(|x| x * half_dt));
                // // p = p_half - (dt / 2) * grad
                for (pv, h) in itertools::izip!(&mut p, &half_dt_grad) {
                    *pv -= h;  // p here is indeed p
                }

                let k = kinetic_energy(&p, &diag_m_inv);
                let h = k + u;

                for (qvd, qv) in itertools::izip!(&mut qs[*i], &q) {
                    *qvd = *qv;
                }

                for (pvd, pv) in itertools::izip!(&mut ps[*i], &p) {
                    *pvd = *pv;
                }

                log_pi[*i] = -h;

                while check_i < checks.len() {
                    let (left, right) = checks[check_i];
                    if right < leapfrogs_taken {
                        let (start, end) = if directions[depth] > 0 { (left, right) } else { (depth_steps.len() - left - 1, depth_steps.len() - right - 1) };
                        let start_i = depth_steps[start];
                        let end_i = depth_steps[end];
    
                        let is_uturn = uturn(
                            &qs[end_i],
                            &qs[start_i],
                            &ps[end_i],
                            &ps[start_i],
                        );
    
                        if is_uturn {
                            uturn_detected_new_tree = true;
                            break;
                        }

                        check_i += 1;
                    } else {
                        break;
                    }
                }

                if uturn_detected_new_tree {
                    break;
                }
            }

            // Merging the two trees requires one more uturn check from the overall left to right states
            let uturn_detected = uturn_detected_new_tree || uturn(&qs[i_right], &qs[i_left], &ps[i_right], &ps[i_left]);

            // Accept statistic from ordinary HMC
            // Only compute the accept probability for the steps done
            let mut p_tree_accept = 0.0;
            let log_pi_steps = Vec::from_iter((0..leapfrogs_taken).map(|i| log_pi[depth_steps[i]]));
            for i in 0..leapfrogs_taken {
                let energy_loss = h0 + log_pi_steps[i];
                p_tree_accept += energy_loss.exp().min(1.0) / leapfrogs_taken as f64;

                if energy_loss.is_nan() || energy_loss.abs() > 1000.0 {
                    divergence = true;
                }
            }

            // Divergence
            if divergence {
                if accept_stat.is_none() {
                    accept_stat = Some(0.0);
                }

                break;
            }

            if uturn_detected {
                // If we u-turn on the first step, grab something for the accept_stat
                if accept_stat.is_none() {
                    accept_stat = Some(p_tree_accept);
                }

                break
            }

            let old_accept_stat = accept_stat;

            accept_stat = match accept_stat {
                None => Some(p_tree_accept),
                Some(accept_stat) => {
                    let weight = leapfrogs_taken as f64 / (leapfrogs_taken + depth_steps.len() - 1) as f64;
                    Some(accept_stat * (1.0 - weight) + p_tree_accept * weight)
                }
            };

            // log of the sum of pi (A.3.1 of https://arxiv.org/abs/1701.02434) of the new subtree
            let log_sum_pi_new = log_sum_exp(&log_pi_steps);

            // sample from the new subtree according to the equation in A.2.1 in https://arxiv.org/abs/1701.02434
            //  (near the end of that section)
            let i_new = if depth > 1 {
                let probs = softmax(&log_pi_steps);

                let choice_draw = choice_draws[depth - 1];

                let mut total_prob = 0.0_f64;
                let mut depth_step : usize = 0;
                for i in 0..probs.len() {
                    let next_prob = total_prob + probs[i];
                    depth_step = depth_steps[i];
                    if choice_draw >= total_prob && choice_draw < next_prob {
                        break;
                    } else {
                        total_prob = next_prob;
                    }
                }
                depth_step
            } else {
                depth_steps[0]
            };

            // Pick between the samples generated from the new and old subtrees using the biased progressive sampling in
            //  A.3.2 of https://arxiv.org/abs/1701.02434
            let p_new = (log_sum_pi_new - log_sum_pi_old).exp().min(1.0);
            if pick_draws[depth - 1] < p_new {
                i_old = i_new
            }

            // Update log of sum of pi of overall tree
            log_sum_pi_old = log_sum_exp(&vec![log_sum_pi_old, log_sum_pi_new]);

        }

        // Get the final sample
        let q = qs[i_old].clone();
            
        Ok(())
    })
    //println!("Hello, world!");
}
