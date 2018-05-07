use std::time::Instant;

pub struct Profiler {
    start_time: Instant,
    curr: Option<Step>,
}

struct Step {
    start_time: Instant,
    name: String,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            curr: None,
        }
    }

    pub fn step(&mut self, step_name: &str) {
        self.finish();
        println!("{} - START", step_name);
        self.curr = Some(Step {
            start_time: Instant::now(),
            name: step_name.to_string(),
        });
    }

    pub fn finish(&mut self) {
        if let Some(ref step) = self.curr {
            let elapsed = step.start_time.elapsed();
            println!(
                "{} - STOP {:.3} s",
                step.name,
                elapsed.as_secs() as f64 + elapsed.subsec_millis() as f64 / 1_000.0,
            );
        }
        self.curr = None
    }

    pub fn total(&mut self) {
        self.finish();
        let elapsed = self.start_time.elapsed();
        println!(
            "TOTAL - {:.3} s",
            elapsed.as_secs() as f64 + elapsed.subsec_millis() as f64 / 1_000.0,
        );
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        self.total()
    }
}
