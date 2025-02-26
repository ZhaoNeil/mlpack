#ifndef MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP

#include <mlpack/core.hpp>

namespace mlpack {

class Serverless {
   public:
    class State {
       public:
        State() : data(nMetrics, nCores) { /* nothing to do here */ }

        /**
         * Construct a state instance from given data.
         *
         * @param data Data for the state.
         */
        // State(const arma::mat& data) : data(data) { /* Nothing to do here */
        // }
        State(const arma::mat& inputData) : data(inputData) {
            if (data.is_empty()) {
                throw std::runtime_error(
                    "State initialization error: data is empty.");
            }
        }

        inline size_t FlattenIndex(size_t row, size_t col) const {
            // row ∈ [0,13], col ∈ [0,59]
            // nRows = 14
            return row + col * 14;
        }

        // The historical response time (sum or average) of tasks completed on
        // this core.
        double TaskRespondTime(size_t core) const {
            size_t idx = FlattenIndex(0, core);
            return data(idx);
        }
        double& TaskRespondTime(size_t core) {
            size_t idx = FlattenIndex(0, core);
            return data(idx);
        }

        // The historical execution time (sum or average) of tasks completed on
        // this core.
        double TaskExecTime(size_t core) const {
            size_t idx = FlattenIndex(1, core);
            return data(idx);
        }
        double& TaskExecTime(size_t core) {
            size_t idx = FlattenIndex(1, core);
            return data(idx);
        }

        // The sum of on CPU time of tasks in the current individual queue. If
        // no migration, then no need for this metric.
        double TaskCPUtime(size_t core) const {
            size_t idx = FlattenIndex(2, core);
            return data(idx);
        }
        double& TaskCPUtime(size_t core) {
            size_t idx = FlattenIndex(2, core);
            return data(idx);
        }

        // Memory usage of running tasks on this core.
        // This metric is questionable. No need for one-time scheduling, i.e. no
        // migration for the current running tasks. But it might be useful for
        // the rescheduling of the preempted tasks.
        double TaskMemory(size_t core) const {
            size_t idx = FlattenIndex(3, core);
            return data(idx);
        }
        double& TaskMemory(size_t core) {
            size_t idx = FlattenIndex(3, core);
            return data(idx);
        }

        // number of preemptions happened on this core
        double PreemptCountPerCore(size_t core) const {
            size_t idx = FlattenIndex(4, core);
            return data(idx);
        }
        double& PreemptCountPerCore(size_t core) {
            size_t idx = FlattenIndex(4, core);
            return data(idx);
        }

        // current CPU uer time on this core
        double CPU_user_time(size_t core) const {
            size_t idx = FlattenIndex(5, core);
            return data(idx);
        }
        double& CPU_user_time(size_t core) {
            size_t idx = FlattenIndex(5, core);
            return data(idx);
        }

        // current CPU nice time on this core
        double CPU_nice_time(size_t core) const {
            size_t idx = FlattenIndex(6, core);
            return data(idx);
        }
        double& CPU_nice_time(size_t core) {
            size_t idx = FlattenIndex(6, core);
            return data(idx);
        }

        // current CPU system time on this core
        double CPU_system_time(size_t core) const {
            size_t idx = FlattenIndex(7, core);
            return data(idx);
        }
        double& CPU_system_time(size_t core) {
            size_t idx = FlattenIndex(7, core);
            return data(idx);
        }

        // current CPU idle time on this core
        double CPU_idle_time(size_t core) const {
            size_t idx = FlattenIndex(8, core);
            return data(idx);
        }
        double& CPU_idle_time(size_t core) {
            size_t idx = FlattenIndex(8, core);
            return data(idx);
        }

        // current CPU iowait time on this core
        double CPU_iowait_time(size_t core) const {
            size_t idx = FlattenIndex(9, core);
            return data(idx);
        }
        double& CPU_iowait_time(size_t core) {
            size_t idx = FlattenIndex(9, core);
            return data(idx);
        }

        // current CPU irq time on this core
        double CPU_irq_time(size_t core) const {
            size_t idx = FlattenIndex(10, core);
            return data(idx);
        }
        double& CPU_irq_time(size_t core) {
            size_t idx = FlattenIndex(10, core);
            return data(idx);
        }

        // current CPU softirq time on this core
        double CPU_softirq_time(size_t core) const {
            size_t idx = FlattenIndex(11, core);
            return data(idx);
        }
        double& CPU_softirq_time(size_t core) {
            size_t idx = FlattenIndex(11, core);
            return data(idx);
        }

        // current CPU steal time on this core
        double CPU_steal_time(size_t core) const {
            size_t idx = FlattenIndex(12, core);
            return data(idx);
        }
        double& CPU_steal_time(size_t core) {
            size_t idx = FlattenIndex(12, core);
            return data(idx);
        }

        // current CPU queue length on this core
        double CPU_queue_length(size_t core) const {
            size_t idx = FlattenIndex(13, core);
            return data(idx);
        }
        double& CPU_queue_length(size_t core) {
            size_t idx = FlattenIndex(13, core);
            return data(idx);
        }

        // double GetMetricValue(size_t metricIndex, size_t coreIndex) const {
        //     return data(metricIndex, coreIndex);
        // }

        arma::rowvec TaskResponseTime() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec TaskResponseTime() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec TaskExecTime() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec TaskExecTime() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec TaskCPUtime() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec TaskCPUtime() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec TaskMemory() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec TaskMemory() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec PreemptCountPerCore() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec PreemptCountPerCore() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_user_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_user_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_nice_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_nice_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_system_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_system_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_idle_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_idle_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_iowait_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_iowait_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_irq_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_irq_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_softirq_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_softirq_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_steal_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_steal_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        arma::rowvec CPU_queue_length() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }
        arma::rowvec CPU_queue_length() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(0);
            return r;
        }

        // const arma::rowvec& GetMetricRow(size_t metricIndex) const {
        //     if (metricIndex >= dimension) {
        //         throw std::invalid_argument(
        //             "Invalid metric index. Must be less than " + dimension);
        //     }
        //     return data.row(metricIndex);
        // }

        // Set a metric's value
        // void SetMetricValue(size_t metricIndex, size_t coreIndex,
        //                     double value) {
        //     data(metricIndex, coreIndex) = value;
        // }

        // void UpdateMetrics(const arma::colvec& newMetrics) {
        //     if (newMetrics.n_elem != dimension) {
        //         throw std::invalid_argument(
        //             "Input metrics size does not match the state
        //             dimension.");
        //     }
        //     data = arma::reshape(newMetrics, 14, 60);
        // }

        arma::mat& Data() { return data; }
        const arma::mat& Data() const { return data; }

        //! Encode the state to a column vector.
        arma::colvec Encode() {
            if (data.is_empty()) {
                throw std::runtime_error(
                    "Encode() error: State data is empty!");
            }
            arma::colvec flattenedState = arma::vectorise(data);
            std::cout << "Flatten state size = " << flattenedState.n_elem
                      << std::endl;
            return flattenedState;
        }

        static constexpr size_t dimension = 14 * 60;

        static constexpr size_t nCores = 60;

        static constexpr size_t nMetrics = 14;

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
    Serverless(const size_t maxSteps, const double doneReward,
               const arma::mat& inputData, size_t inputMetrics,
               size_t inputCores)
        : maxSteps(maxSteps),
          doneReward(doneReward),
          serverlessData(inputData),
          serverlessMetrics(inputMetrics),
          serverlessCores(inputCores),
          stepsPerformed(0) {
        std::cout << "Serverless initialized with data size: "
                  << "rows=" << serverlessData.n_rows
                  << ", cols=" << serverlessData.n_cols << std::endl;
    }

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
        std::cout << "dest_core=" << dest_core << std::endl;
        double maxTask = state.CPU_queue_length().max();
        double minTask = state.CPU_queue_length().min();
        std::cout << "maxTask=" << maxTask << ", minTask=" << minTask
                  << std::endl;
        std::cout << "state.CPU_queue_length(dest_core)="
                  << state.CPU_queue_length(dest_core) << std::endl;
        if (state.CPU_queue_length(dest_core) == maxTask) {
            std::cout << "enter if" << std::endl;
            return -1.0;
        } else if (state.CPU_queue_length(dest_core) == minTask) {
            std::cout << "enter else if" << std::endl;
            return 1.0;
        }
        std::cout << "test" << std::endl;
        // corresponding core CPU_queue_length add 1
        // nextState.CPU_queue_length(dest_core) =
        //     state.CPU_queue_length(dest_core) + 1;

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
        std::cout << "In Serverless: data.n_rows=" << serverlessData.n_rows
                  << ", data.n_cols=" << serverlessData.n_cols << std::endl;
        stepsPerformed = 0;
        return State(serverlessData);
    }

    /**
     * This function checks if the serverless has reached the terminal state.
     *
     * @param state The current state.
     * @return true if state is a terminal state, otherwise false.
     */
    bool IsTerminal(const State& state) const {
        if (maxSteps != 0 && stepsPerformed >= maxSteps) {
            std::cout << "Episode terminated due to the maximum number of steps"
                      << "being taken." << std::endl;
            return true;
        } else if (arma::all(state.CPU_queue_length() == 0.0)) {
            std::cout << "Episode terminated due all tasks finished."
                      << std::endl;
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
    arma::mat serverlessData;

    // Dimension of the metrics
    size_t serverlessMetrics;

    // Dimension of the number of cores
    size_t serverlessCores;
};
}  // namespace mlpack

#endif