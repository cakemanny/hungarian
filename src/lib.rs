/* Copyright (c) 2012 Kevin L. Stern
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Ported to JavaScript by Gerard Meier. For simplicity, it's a literal port,
 * omitting any typical JavaScript idioms & paradigms. For usage examples see
 * the Test() function at the bottom.
 *
 * Ported to Rust by Daniel Golding.
 */
/*
 * https://github.com/KevinStern/software-and-algorithms/blob/master/src/main/java/blogspot/software_and_algorithms/stern_library/optimization/HungarianAlgorithm.java
 * https://github.com/Gerjo/esoteric/blob/master/Hungarian.js
*/
use std::cmp;
use std::result::{Result};
use std::f64::INFINITY;

#[derive(Debug, PartialEq)]
pub struct HungarianAlgorithm {
    cost_matrix: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
    dim: usize,
    label_by_worker: Vec<f64>,
    label_by_job: Vec<f64>,
    min_slack_worker_by_job: Vec<i32>,
    min_slack_value_by_job: Vec<f64>,
    committed_workers: Vec<bool>,
    parent_worker_by_committed_job: Vec<i32>,
    match_job_by_worker: Vec<i32>,
    match_worker_by_job: Vec<i32>,
}

/**
 * An implementation of the Hungarian algorithm for solving the assignment
 * problem. An instance of the assignment problem consists of a number of
 * workers along with a number of jobs and a cost matrix which gives the cost of
 * assigning the i'th worker to the j'th job at position (i, j). The goal is to
 * find an assignment of workers to jobs so that no job is assigned more than
 * one worker and so that no worker is assigned to more than one job in such a
 * manner so as to minimize the total cost of completing the jobs.
 * <p>
 *
 * An assignment for a cost matrix that has more workers than jobs will
 * necessarily include unassigned workers, indicated by an assignment value of
 * -1; in no other circumstance will there be unassigned workers. Similarly, an
 * assignment for a cost matrix that has more jobs than workers will necessarily
 * include unassigned jobs; in no other circumstance will there be unassigned
 * jobs. For completeness, an assignment for a square cost matrix will give
 * exactly one unique worker to each job.
 *
 * This version of the Hungarian algorithm runs in time O(n^3), where n is the
 * maximum among the number of workers and the number of jobs.
 *
 * @author Kevin L. Stern
 */
impl HungarianAlgorithm {

    /**
     * Construct an instance of the algorithm.
     *
     * @param cost_matrix
     *          the cost matrix, where matrix[i][j] holds the cost of assigning
     *          worker i to job j, for all i, j. The cost matrix must not be
     *          irregular in the sense that all rows must be the same length; in
     *          addition, all entries must be non-infinite numbers.
     */
    pub fn new(cost_matrix: Vec<Vec<f64>>) -> Result<HungarianAlgorithm, String> {
        if cost_matrix.len() == 0 {
            return Err(String::from("cost_matrix must be non-empty"));
        }
        if cost_matrix[0].len() == 0 {
            return Err(String::from("cost_matrix must be non-empty"));
        }
        let rows = cost_matrix.len();
        let cols = cost_matrix[0].len();
        let dim = cmp::max(rows, cols);

        let mut our_cost_matrix: Vec<Vec<f64>> = vec![vec![ 0.0; dim ]; dim];

        for w in 0 .. dim {
            if w < rows {
                let row = &cost_matrix[w];
                if row.len() != cols {
                    return Err(String::from("Irregular cost matrix"))
                }
                for j in 0 .. cols {
                    if row[j].is_infinite() {
                        return Err(String::from("Infinite cost"))
                    }
                    if row[j].is_nan() {
                        return Err(String::from("NaN cost"))
                    }
                }
                for j in 0 .. cols {
                    our_cost_matrix[w][j] = row[j]
                }
            }
        }

        let algo = HungarianAlgorithm {
            cost_matrix: our_cost_matrix,
            rows,
            cols,
            dim,
            label_by_worker: vec![0.0; dim],
            label_by_job: vec![0.0; dim],
            min_slack_worker_by_job: vec![0; dim],
            min_slack_value_by_job: vec![0.0; dim],
            committed_workers: vec![false; dim],
            parent_worker_by_committed_job: vec![0; dim],
            match_job_by_worker: vec![-1; dim],
            match_worker_by_job: vec![-1; dim],
        };
        Ok(algo)
    }

    /**
     * Compute an initial feasible solution by assigning zero labels to the
     * workers and by assigning to each job a label equal to the minimum cost
     * among its incident edges.
     */
    fn compute_initial_feasible_solution(&mut self) {
        for j in 0 .. self.dim {
            self.label_by_job[j] = INFINITY;
        }
        for w in 0 .. self.dim {
            for j in 0 .. self.dim {
                if self.cost_matrix[w][j] < self.label_by_job[j] {
                    self.label_by_job[j] = self.cost_matrix[w][j];
                }
            }
        }
    }

    /**
     * Execute the algorithm.
     *
     * @return the minimum cost matching of workers to jobs based upon the
     *         provided cost matrix. A matching value of -1 indicates that the
     *         corresponding worker is unassigned.
     */
    pub fn execute(&mut self) -> Vec<i32> {
        /*
         * Heuristics to improve performance: Reduce rows and columns by their
         * smallest element, compute an initial non-zero dual feasible solution and
         * create a greedy matching from workers to jobs of the cost matrix.
         */
        self.reduce();
        self.compute_initial_feasible_solution();
        self.greedy_match();

        let mut w = self.fetch_unmatched_worker();
        while w < self.dim {
            self.initialize_phase(w);
            self.execute_phase();
            w  = self.fetch_unmatched_worker();
        }
        let mut result = self.match_job_by_worker[0 .. self.rows].to_vec();
        for w in 0 .. result.len() {
            if result[w] >= self.cols as i32 {
                result[w] = -1;
            }
        };
        result
    }

    /**
     * Execute a single phase of the algorithm. A phase of the Hungarian algorithm
     * consists of building a set of committed workers and a set of committed jobs
     * from a root unmatched worker by following alternating unmatched/matched
     * zero-slack edges. If an unmatched job is encountered, then an augmenting
     * path has been found and the matching is grown. If the connected zero-slack
     * edges have been exhausted, the labels of committed workers are increased by
     * the minimum slack among committed workers and non-committed jobs to create
     * more zero-slack edges (the labels of committed jobs are simultaneously
     * decreased by the same amount in order to maintain a feasible labeling).
     * <p>
     *
     * The runtime of a single phase of the algorithm is O(n^2), where n is the
     * dimension of the internal square cost matrix, since each edge is visited at
     * most once and since increasing the labeling is accomplished in time O(n) by
     * maintaining the minimum slack values among non-committed jobs. When a phase
     * completes, the matching will have increased in size.
     */
    fn execute_phase(&mut self) {
        loop {
            let mut min_slack_worker: i32 = -1;
            let mut min_slack_job: i32 = -1;
            let mut min_slack_value = INFINITY;
            for j in 0 .. self.dim {
                if self.parent_worker_by_committed_job[j] == -1 {
                    if self.min_slack_value_by_job[j] < min_slack_value {
                        min_slack_value = self.min_slack_value_by_job[j];
                        min_slack_worker = self.min_slack_worker_by_job[j];
                        min_slack_job = j as i32;
                    }
                }
            }
            assert!(min_slack_job >= 0);
            assert!(min_slack_worker >= 0);

            if min_slack_value > 0.0 {
                self.update_labeling(min_slack_value);
            }
            self.parent_worker_by_committed_job[min_slack_job as usize] =
                min_slack_worker;
            if self.match_worker_by_job[min_slack_job as usize] == -1 {
                /*
                 * An augmenting path has been found.
                 */
                let mut commited_job = min_slack_job;
                let mut parent_worker =
                    self.parent_worker_by_committed_job[commited_job as usize];
                loop {
                    let temp = self.match_job_by_worker[parent_worker as usize];
                    assert!(parent_worker > 0 && commited_job > 0);
                    self.match_(parent_worker as usize, commited_job as usize);
                    commited_job = temp;
                    if commited_job == -1 {
                        break
                    }
                    parent_worker = self.parent_worker_by_committed_job[commited_job as usize];
                }
                return;
            } else {
                /*
                 * Update slack values since we increased the size of the committed
                 * workers set.
                 */
                let worker = self.match_worker_by_job[min_slack_job as usize];
                self.committed_workers[worker as usize] = true;
                for j in 0 .. self.dim {
                    if self.parent_worker_by_committed_job[j] == -1 {
                        let slack: f64 = self.cost_matrix[worker as usize][j]
                            - self.label_by_worker[worker as usize]
                            - self.label_by_job[j];
                        if self.min_slack_value_by_job[j] > slack {
                            self.min_slack_value_by_job[j] = slack;
                            self.min_slack_worker_by_job[j] = worker;
                        }
                    }
                }
            }
        }
    }

    /**
     *
     * @return the first unmatched worker or {@link #dim} if none.
     */
    fn fetch_unmatched_worker(&self) -> usize {
        for w in 0 .. self.dim {
            if self.match_job_by_worker[w] == -1 {
                return w;
            }
        }
        return self.dim;
    }

    /**
     * Find a valid matching by greedily selecting among zero-cost matchings.
     * This is a heuristic to jump-start the augmentation algorithm.
     */
    fn greedy_match(&mut self) {
        for w in 0 .. self.dim {
            for j in 0 .. self.dim {
                if self.match_job_by_worker[w] == -1
                    && self.match_worker_by_job[j] == -1
                    && self.cost_matrix[w][j] - self.label_by_worker[w]
                        - self.label_by_job[j] == 0.0
                {
                    self.match_(w, j);
                }
            }
        }
    }

    /**
     * Initialize the next phase of the algorithm by clearing the committed
     * workers and jobs sets and by initializing the slack arrays to the values
     * corresponding to the specified root worker.
     *
     * @param w
     *          the worker at which to root the next phase.
     */
    fn initialize_phase(&mut self, w: usize) {
        self.committed_workers.copy_from_slice(&vec![false; self.dim]);
        self.parent_worker_by_committed_job.copy_from_slice(&vec![-1; self.dim]);
        for j in 0 .. self.dim {
            self.min_slack_value_by_job[j] = self.cost_matrix[w][j]
                - self.label_by_worker[w] - self.label_by_job[j];
            self.min_slack_worker_by_job[j] = w as i32;
        }
    }

    /**
     * Helper method to record a matching between worker w and job j.
     */
    fn match_(&mut self, w: usize, j: usize) {
        assert!(w <= self.dim);
        assert!(j <= self.dim);
        self.match_job_by_worker[w] = j as i32;
        self.match_worker_by_job[j] = w as i32;
    }

    /**
     * Reduce the cost matrix by subtracting the smallest element of each row from
     * all elements of the row as well as the smallest element of each column from
     * all elements of the column. Note that an optimal assignment for a reduced
     * cost matrix is optimal for the original cost matrix.
     */
    fn reduce(&mut self) {
        for w in 0 .. self.dim {
            let mut min = INFINITY;
            for j in 0 .. self.dim {
                if self.cost_matrix[w][j] < min {
                    min = self.cost_matrix[w][j];
                }
            }
            for j in 0 .. self.dim {
                self.cost_matrix[w][j] -= min;
            }
        }
        let mut min = vec![INFINITY; self.dim];
        for w in 0 .. self.dim {
            for j in 0 .. self.dim {
                if self.cost_matrix[w][j] < min[j] {
                    min[j] = self.cost_matrix[w][j];
                }
            }
        }
        for w in 0 .. self.dim {
            for j in 0 .. self.dim {
                self.cost_matrix[w][j] -= min[j];
            }
        }
    }

    /**
     * Update labels with the specified slack by adding the slack value for
     * committed workers and by subtracting the slack value for committed jobs. In
     * addition, update the minimum slack values appropriately.
     */
    fn update_labeling(&mut self, slack: f64) {
        for w in 0 .. self.dim {
            if self.committed_workers[w] {
                self.label_by_worker[w] += slack;
            }
        }
        for j in 0 .. self.dim {
            if self.parent_worker_by_committed_job[j] != -1 {
                self.label_by_job[j] -= slack;
            } else {
                self.min_slack_value_by_job[j] -= slack;
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use crate::HungarianAlgorithm;
    use std::f64::INFINITY;

    #[test]
    fn new_works() {
        let costs: Vec<Vec<f64>> = vec![
            vec![2.0, 3.0, 3.0],
            vec![3.0, 2.0, 3.0],
            vec![3.0, 3.0, 2.0],
        ];
        let r = HungarianAlgorithm::new(costs);

        assert_eq!(r, Ok(HungarianAlgorithm {
            cost_matrix: vec![
                vec![2.0, 3.0, 3.0],
                vec![3.0, 2.0, 3.0],
                vec![3.0, 3.0, 2.0],
            ],
            rows: 3,
            cols: 3,
            dim: 3,
            label_by_worker: vec![0.0, 0.0, 0.0],
            label_by_job: vec![0.0, 0.0, 0.0],
            min_slack_worker_by_job: vec![0, 0, 0],
            min_slack_value_by_job: vec![0.0, 0.0, 0.0],
            committed_workers: vec![false, false, false],
            parent_worker_by_committed_job: vec![0, 0, 0],
            match_job_by_worker: vec![-1, -1, -1],
            match_worker_by_job: vec![-1, -1, -1],
        }));
    }

    #[test]
    fn error_on_irregular_size() {
        let costs: Vec<Vec<f64>> = vec![
            vec![2.0, 3.0, 3.0],
            vec![3.0, 2.0],
            vec![3.0, 3.0, 2.0],
        ];
        let r = HungarianAlgorithm::new(costs);

        assert_eq!(r, Err(String::from("Irregular cost matrix")));
    }

    #[test]
    fn error_on_infinite_cost() {
        let costs: Vec<Vec<f64>> = vec![
            vec![2.0, 3.0, 3.0],
            vec![3.0, 2.0, INFINITY],
            vec![3.0, 3.0, 2.0],
        ];
        let r = HungarianAlgorithm::new(costs);

        assert_eq!(r, Err(String::from("Infinite cost")));
    }

    #[test]
    fn execute_works() {
        // the example from the top of the wikipedia page
        let costs: Vec<Vec<f64>> = vec![
            vec![2.0, 3.0, 3.0],
            vec![3.0, 2.0, 3.0],
            vec![3.0, 3.0, 2.0],
        ];
        let r = HungarianAlgorithm::new(costs);
        let mut algo = r.unwrap();

        let result = algo.execute();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn more_cases() {
        fn assert_exec(costs: Vec<Vec<f64>>, exp_result: Vec<i32>) {
            let mut algo = HungarianAlgorithm::new(costs).unwrap();
            let result = algo.execute();
            assert_eq!(result, exp_result);
        }

        assert_exec(vec![
            vec![1.0],
        ], vec![0]);
        assert_exec(vec![
            vec![1.0],
            vec![1.0],
        ], vec![0, -1]);
        assert_exec(vec![
            vec![1.0, 1.0],
        ], vec![0]);
        assert_exec(vec![
               vec![1.0, 1.0],
               vec![1.0, 1.0],
        ], vec![0, 1]);
        assert_exec(vec![
               vec![1.0, 1.0],
               vec![1.0, 1.0],
               vec![1.0, 1.0],
        ], vec![0, 1, -1]);
        assert_exec(vec![
               vec![1.0, 2.0, 3.0],
               vec![6.0, 5.0, 4.0],
        ], vec![0, 2]);
        assert_exec(vec![
               vec![1.0, 2.0, 3.0],
               vec![6.0, 5.0, 4.0],
               vec![1.0, 1.0, 1.0],
        ], vec![0, 2, 1]);
        assert_exec(vec![
               vec![10.0, 25.0, 15.0, 20.0],
               vec![15.0, 30.0,  5.0, 15.0],
               vec![35.0, 20.0, 12.0, 24.0],
               vec![17.0, 25.0, 24.0, 20.0],
        ], vec![0, 2, 1, 3]);
    }

}
