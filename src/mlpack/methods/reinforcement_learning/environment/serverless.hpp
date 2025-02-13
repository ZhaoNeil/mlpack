#ifndef MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP

#include <mlpack/core.hpp>

namespace mlpack {

class Serverless {
   public:
    class State {
       public:
        State()
            : data(dimension, nCores,
                   arma::fill::zeros) { /* nothing to do here */ }

        /**
         * Construct a state instance from given data.
         *
         * @param data Data for the state.
         */
        State(const arma::mat& data) : data(data) { /* Nothing to do here */ }

        // The historical response time (sum or average) of tasks completed on
        // this core.
        double TaskRespondTime(size_t core) const { return data(0, core); }
        double& TaskRespondTime(size_t core) { return data(0, core); }

        // The historical execution time (sum or average) of tasks completed on
        // this core.
        double TaskExecTime(size_t core) const { return data(1, core); }
        double& TaskExecTime(size_t core) { return data(1, core); }

        // The sum of on CPU time of tasks in the current individual queue. If
        // no migration, then no need for this metric.
        double TaskCPUtime(size_t core) const { return data(2, core); }
        double& TaskCPUtime(size_t core) { return data(2, core); }

        // Memory usage of running tasks on this core.
        // This metric is questionable. No need for one-time scheduling, i.e. no
        // migration for the current running tasks. But it might be useful for
        // the rescheduling of the preempted tasks.
        double TaskMemory(size_t core) const { return data(3, core); }
        double& TaskMemory(size_t core) { return data(3, core); }

        // number of preemptions happened on this core
        double PreemptCountPerCore(size_t core) const { return data(4, core); }
        double& PreemptCountPerCore(size_t core) { return data(4, core); }

        // current CPU uer time on this core
        double CPU_user_time(size_t core) const { return data(5, core); }
        double& CPU_user_time(size_t core) { return data(5, core); }

        // current CPU nice time on this core
        double CPU_nice_time(size_t core) const { return data(6, core); }
        double& CPU_nice_time(size_t core) { return data(6, core); }

        // current CPU system time on this core
        double CPU_system_time(size_t core) const { return data(7, core); }
        double& CPU_system_time(size_t core) { return data(7, core); }

        // current CPU idle time on this core
        double CPU_idle_time(size_t core) const { return data(8, core); }
        double& CPU_idle_time(size_t core) { return data(8, core); }

        // current CPU iowait time on this core
        double CPU_iowait_time(size_t core) const { return data(9, core); }
        double& CPU_iowait_time(size_t core) { return data(9, core); }

        // current CPU irq time on this core
        double CPU_irq_time(size_t core) const { return data(10, core); }
        double& CPU_irq_time(size_t core) { return data(10, core); }

        // current CPU softirq time on this core
        double CPU_softirq_time(size_t core) const { return data(11, core); }
        double& CPU_softirq_time(size_t core) { return data(11, core); }

        // current CPU steal time on this core
        double CPU_steal_time(size_t core) const { return data(12, core); }
        double& CPU_steal_time(size_t core) { return data(12, core); }

        // current CPU queue length on this core
        double CPU_queue_length(size_t core) const { return data(13, core); }
        double& CPU_queue_length(size_t core) { return data(13, core); }

        double GetMetricValue(size_t metricIndex, size_t coreIndex) const {
            return data(metricIndex, coreIndex);
        }

        const arma::subview_row<double> TaskResponseTime() const {
            return data.row(0);
        }
        arma::subview_row<double> TaskResponseTime() { return data.row(0); }

        const arma::subview_row<double> TaskExecTime() const {
            return data.row(1);
        }
        arma::subview_row<double> TaskExecTime() { return data.row(1); }

        const arma::subview_row<double> TaskCPUtime() const {
            return data.row(2);
        }
        arma::subview_row<double> TaskCPUtime() { return data.row(2); }

        const arma::subview_row<double> TaskMemory() const {
            return data.row(3);
        }
        arma::subview_row<double> TaskMemory() { return data.row(3); }

        const arma::subview_row<double> PreemptCountPerCore() const {
            return data.row(4);
        }
        arma::subview_row<double> PreemptCountPerCore() { return data.row(4); }

        const arma::subview_row<double> CPU_user_time() const {
            return data.row(5);
        }
        arma::subview_row<double> CPU_user_time() { return data.row(5); }

        const arma::subview_row<double> CPU_nice_time() const {
            return data.row(6);
        }
        arma::subview_row<double> CPU_nice_time() { return data.row(6); }

        const arma::subview_row<double> CPU_system_time() const {
            return data.row(7);
        }
        arma::subview_row<double> CPU_system_time() { return data.row(7); }

        const arma::subview_row<double> CPU_idle_time() const {
            return data.row(8);
        }
        arma::subview_row<double> CPU_idle_time() { return data.row(8); }

        const arma::subview_row<double> CPU_iowait_time() const {
            return data.row(9);
        }
        arma::subview_row<double> CPU_iowait_time() { return data.row(9); }

        const arma::subview_row<double> CPU_irq_time() const {
            return data.row(10);
        }
        arma::subview_row<double> CPU_irq_time() { return data.row(10); }

        const arma::subview_row<double> CPU_softirq_time() const {
            return data.row(11);
        }
        arma::subview_row<double> CPU_softirq_time() { return data.row(11); }

        const arma::subview_row<double> CPU_steal_time() const {
            return data.row(12);
        }
        arma::subview_row<double> CPU_steal_time() { return data.row(12); }

        const arma::subview_row<double> CPU_queue_length() const {
            return data.row(13);
        }
        arma::subview_row<double> CPU_queue_length() { return data.row(13); }

        const arma::rowvec& GetMetricRow(size_t metricIndex) const {
            if (metricIndex >= dimension) {
                throw std::invalid_argument(
                    "Invalid metric index. Must be less than " + dimension);
            }
            return data.row(metricIndex);
        }

        // Set a metric's value
        void SetMetricValue(size_t metricIndex, size_t coreIndex,
                            double value) {
            data(metricIndex, coreIndex) = value;
        }

        void UpdateMetrics(const arma::colvec& newMetrics) {
            if (newMetrics.n_elem != dimension) {
                throw std::invalid_argument(
                    "Input metrics size does not match the state dimension.");
            }
            data = newMetrics;
        }

        arma::mat& Data() { return data; }
        const arma::mat& Data() const { return data; }

        //! Encode the state to a column vector.
        const arma::colvec& Encode() { return arma::vectorise(data); }

        //! Dimension of the metrics
        size_t dimension = GetRows();

        //! Dimension of the number of cores
        size_t nCores = GetCols();

        size_t GetRows() const { return data.n_rows; }

        size_t GetCols() const { return data.n_cols; }

       private:
        //! Locally-stored state data.
        arma::mat data;
    };

    class Action {
       public:
        enum actions {
            allocate_core,
        };
        // To store the action.
        Action::actions action;

        // Track the size of the action space.
        static constexpr size_t size = 1;
    };

    /**
     * Construct a Serverless instance using the given constants.
     *
     * @param maxSteps The maximum number of steps allowed.
     * @param doneReward The reward recieved by the agent on success.
     *
     */
    Serverless(const size_t maxSteps = 500, const double doneReward = 1.0,
               arma::mat& data, size_t dimension = 14, size_t nCores = 60)
        : maxSteps(maxSteps),
          doneReward(doneReward),
          stepsPerformed(0),
          data(data) {}

    /**
     * Dynamics of the Serverless instance. Get reward and next state based on
     * current state and current action.
     *
     * @param state The current state.
     * @param action The current action.
     * @param nextState The next state.
     * @return reward,
     */
    double Sample(const State& state, const Action& action, State& nextState) {
        stepsPerformed++;

        // to do: Update the state based on the action.
        size_t dest_core = action.action;
        double maxTask = state.CPU_queue_length().max();
        double minTask = state.CPU_queue_length().min();
        if (state.CPU_queue_length(dest_core) == maxTask) {
            return -1.0;
        } else if (state.CPU_queue_length(dest_core) == minTask) {
            return 1.0;
        }
        // corresponding core CPU_queue_length add 1
        nextState.CPU_queue_length(dest_core) =
            state.CPU_queue_length(dest_core) + 1;

        // to do: update the metrics of the next state

        bool done = IsTerminal(nextState);

        if (done && maxSteps != 0 && stepsPerformed >= maxSteps) {
            return doneReward;
        }

        return 1.0;
    }

    /**
     * Dynamics of Serverless instance. Get reward based on current state and
     * action.
     *
     * @param state The current state.
     * @param action The current action.
     * @return reward,
     */
    double Sample(const State& state, const Action& action) {
        State nextState;
        return Sample(state, action, nextState);
    }

    /**
     * Initial state representation.
     *
     * @return the dummy state.
     */
    State InitialSample() {
        stepsPerformed = 0;
        return State(data);
    }

    /**
     * This function checks if the serverless has reached the terminal state.
     *
     * @param state The current state.
     * @return true if state is a terminal state, otherwise false.
     */
    bool IsTerminal(const State& state) const {
        if (maxSteps != 0 && stepsPerformed >= maxSteps) {
            Log::Info << "Episode terminated due to the maximum number of steps"
                      << "being taken.";
            return true;
        } else if (arma::all(state.CPU_queue_length() == 0.0)) {
            Log::Info << "Episode terminated due all tasks finished.";
            return true;
        }
        return false;
    }

    // Get the number of steps performed.
    size_t StepsPerformed() const { return stepsPerformed; }

    // Get the maximum number of steps allowed.
    size_t MaxSteps() const { return maxSteps; }
    // Set the maximum number of steps allowed.
    size_t& MaxSteps() { return maxSteps; }

   private:
    // Locally-stored maximum number of steps.
    size_t maxSteps;

    // Locally-stored reward for success.
    double doneReward;

    // Locally-stored number of steps performed.
    size_t stepsPerformed;

    // Locally-stored data.
    arma::mat data;

    // Dimension of the metrics
    size_t dimension = data.n_rows;

    // Dimension of the number of cores
    size_t nCores = data.n_cols;
};
}  // namespace mlpack

#endif