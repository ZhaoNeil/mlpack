#ifndef MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP

#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <unistd.h>

#include <mlpack.hpp>
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
            // std::cout << "TaskResponseTime=" << data.row(0) << std::endl;
            // std::cout << "TaskExecTime=" << data.row(1) << std::endl;
            // std::cout << "TaskCPUtime=" << data.row(2) << std::endl;
            // std::cout << "TaskMemory=" << data.row(3) << std::endl;
            // std::cout << "PreemptCountPerCore=" << data.row(4) << std::endl;
            // std::cout << "CPU_user_time=" << data.row(5) << std::endl;
            // std::cout << "CPU_nice_time=" << data.row(6) << std::endl;
            // std::cout << "CPU_system_time=" << data.row(7) << std::endl;
            // std::cout << "CPU_idle_time=" << data.row(8) << std::endl;
            // std::cout << "CPU_iowait_time=" << data.row(9) << std::endl;
            // std::cout << "CPU_irq_time=" << data.row(10) << std::endl;
            // std::cout << "CPU_softirq_time=" << data.row(11) << std::endl;
            // std::cout << "CPU_steal_time=" << data.row(12) << std::endl;
            // std::cout << "CPU_queue_length=" << data.row(13) << std::endl;
        }

        inline size_t FlattenIndex(size_t row, size_t col) const {
            // row ∈ [0,13], col ∈ [0,59]
            // nRows = 14
            return row + col * 14;
        }

        // The historical response time (sum or average) of tasks completed on
        // this core.
        double TaskResponseTime(size_t core) const {
            size_t idx = FlattenIndex(0, core);
            return data(idx);
        }
        double& TaskResponseTime(size_t core) {
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
            arma::rowvec r = tmp.row(1);
            return r;
        }
        arma::rowvec TaskExecTime() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(1);
            return r;
        }

        arma::rowvec TaskCPUtime() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(2);
            return r;
        }
        arma::rowvec TaskCPUtime() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(2);
            return r;
        }

        arma::rowvec TaskMemory() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(3);
            return r;
        }
        arma::rowvec TaskMemory() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(3);
            return r;
        }

        arma::rowvec PreemptCountPerCore() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(4);
            return r;
        }
        arma::rowvec PreemptCountPerCore() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(4);
            return r;
        }

        arma::rowvec CPU_user_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(5);
            return r;
        }
        arma::rowvec CPU_user_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(5);
            return r;
        }

        arma::rowvec CPU_nice_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(6);
            return r;
        }
        arma::rowvec CPU_nice_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(6);
            return r;
        }

        arma::rowvec CPU_system_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(7);
            return r;
        }
        arma::rowvec CPU_system_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(7);
            return r;
        }

        arma::rowvec CPU_idle_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(8);
            return r;
        }
        arma::rowvec CPU_idle_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(8);
            return r;
        }

        arma::rowvec CPU_iowait_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(9);
            return r;
        }
        arma::rowvec CPU_iowait_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(9);
            return r;
        }

        arma::rowvec CPU_irq_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(10);
            return r;
        }
        arma::rowvec CPU_irq_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(10);
            return r;
        }

        arma::rowvec CPU_softirq_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(11);
            return r;
        }
        arma::rowvec CPU_softirq_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(11);
            return r;
        }

        arma::rowvec CPU_steal_time() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(12);
            return r;
        }
        arma::rowvec CPU_steal_time() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(12);
            return r;
        }

        arma::rowvec CPU_queue_length() const {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(13);
            return r;
        }
        arma::rowvec CPU_queue_length() {
            arma::mat tmp = arma::reshape(data, nMetrics, nCores);
            arma::rowvec r = tmp.row(13);
            return r;
        }

        arma::mat& Data() { return data; }
        const arma::mat& Data() const { return data; }

        //! Encode the state to a column vector.
        arma::colvec Encode() {
            if (data.is_empty()) {
                throw std::runtime_error(
                    "Encode() error: State data is empty!");
            }
            arma::colvec flattenedState = arma::vectorise(data);
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
            size_t action;
        
            Action() : action(0) {}
            Action(size_t id) : action(id) {}
        
            static constexpr size_t size = 60;
    };
        
        
    // class Action {
    //     public:
    //      enum actions {
    //          allocate_core,
    //      };
    //      // To store the action.
    //      Action::actions action;
 
    //      // Track the size of the action space.
    //      static constexpr size_t size = 1;
    //  };

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
          stepsPerformed(0) {}

    arma::mat GetLatestEnvironmentMetrics(const SharedData& sharedData) {
        sharedDataMutex_.Lock();
        arma::mat data = sharedData.getData();
        sharedDataMutex_.Unlock();
        return data;
    }

    double GetScore(const State& state) {
        double responseTime = arma::accu(state.TaskResponseTime());
        // std::cout << "responseTime=" << responseTime << std::endl;
        double execTime = arma::accu(state.TaskExecTime());
        // std::cout << "execTime=" << execTime << std::endl;
        double cpuTime = arma::accu(state.TaskCPUtime());
        // std::cout << "cpuTime=" << cpuTime << std::endl;
        double memoryUsage = arma::accu(state.TaskMemory());
        // std::cout << "memoryUsage=" << memoryUsage << std::endl;
        double preemptions = arma::accu(state.PreemptCountPerCore());
        // std::cout << "preemptions=" << preemptions << std::endl;
        double queueLength = arma::accu(state.CPU_queue_length());
        // std::cout << "queueLength=" << state.CPU_queue_length() << std::endl;

        double userTime = arma::accu(state.CPU_user_time());
        // std::cout << "userTime=" << userTime << std::endl;
        double niceTime = arma::accu(state.CPU_nice_time());
        // std::cout << "niceTime=" << niceTime << std::endl;
        double systemTime = arma::accu(state.CPU_system_time());
        // std::cout << "systemTime=" << systemTime << std::endl;
        double idleTime = arma::accu(state.CPU_idle_time());
        // std::cout << "idleTime=" << idleTime << std::endl;
        double iowaitTime = arma::accu(state.CPU_iowait_time());
        // std::cout << "iowaitTime=" << iowaitTime << std::endl;
        double irqTime = arma::accu(state.CPU_irq_time());
        // std::cout << "irqTime=" << irqTime << std::endl;
        double softirqTime = arma::accu(state.CPU_softirq_time());
        // std::cout << "softirqTime=" << softirqTime << std::endl;
        double stealTime = arma::accu(state.CPU_steal_time());
        // std::cout << "stealTime=" << stealTime << std::endl;

        double busyTime =
            userTime + niceTime + systemTime + irqTime + softirqTime;
        double totalTime = busyTime + idleTime + iowaitTime + stealTime;
        double utilization = (totalTime > 0) ? busyTime / totalTime : 0.0;

        double w1 = 1.0;   // Response time
        double w2 = 1.0;   // Exec time
        double w3 = 0.1;   // CPU time
        double w4 = 0.05;  // Memory
        double w5 = 0.1;   // Preemptions
        double w6 = 0.1;   // Queue length
        double w7 = 0.1;   // CPU utilization

        double score = -(w1 * responseTime + w2 * execTime);
        score -= (w3 * cpuTime + w4 * memoryUsage + w5 * preemptions +
                  w6 * queueLength + w7 * utilization);

        return score;
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
        std::cout << "stepsPerformed= " << stepsPerformed << std::endl;

        size_t dest_core = action.action;
        std::cout << "dest_core=" << dest_core << std::endl;

        arma::mat latestdata = GetLatestEnvironmentMetrics(sharedData);
        // std::cout << "latestdata=" << latestdata << std::endl;
        nextState = State(latestdata);

        double state_score = GetScore(state);
        double next_state_score = GetScore(nextState);
        // std::cout << "state_score=" << state_score << std::endl;
        // std::cout << "next_state_score=" << next_state_score << std::endl;

        bool done = IsTerminal(nextState);
        if (done && maxSteps != 0 && stepsPerformed >= maxSteps) {
            return doneReward;
        }

        if (next_state_score > state_score) {
            std::cout << "Reward for the action." << std::endl;
            return 1.0;}
        else if (next_state_score < state_score) {
            std::cout << "Penalty for the action." << std::endl;
            return -1.0;}
        else
            std::cout << "No change in score." << std::endl;
            return 0.0;
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
        std::cout << "InitialSample()" << std::endl;
        stepsPerformed = 0;
        arma::mat currentdata = GetLatestEnvironmentMetrics(sharedData);
        // std::cout << "currentdata=" << currentdata << std::endl;
        return State(currentdata);
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