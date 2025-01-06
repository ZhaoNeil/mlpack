#ifndef MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_SERVERLESS_HPP

#include <mlpack/core.hpp>

namespace mlpack {

class Serverless {
   public:
    class state {
       public:
        State() : data(dimension) { /* nothing to do here */ }

        /**
         * Construct a state instance from given data.
         *
         * @param data Data for the state.
         */
        State(const arma::colvec& data)
            : data(data) { /* Nothing to do here */ }

        //! Modify the internal representation of the state.
        arma::colvec& Data() { return data; }

        double TaskOnCore() const { return data[0]; }
        double& TaskOnCore() { return data[0]; }

        double TaskRespondTime() const { return data[1]; }
        double& TaskRespondTime() { return data[1]; }

        double TaskExecTime() const { return data[2]; }
        double& TaskExecTime() { return data[2]; }

        double TaskCPUtime() const { return data[3]; }
        double& TaskCPUtime() { return data[3]; }

        double TaskMemory() const { return data[4]; }
        double& TaskMemory() { return data[4]; }

        double TaskPreemptions() const { return data[5]; }
        double& TaskPreemptions() { return data[5]; }

        double CPU_user_time() const { return data[6]; }
        double& CPU_user_time() { return data[6]; }

        double CPU_nice_time() const { return data[7]; }
        double& CPU_nice_time() { return data[7]; }

        double CPU_system_time() const { return data[8]; }
        double& CPU_system_time() { return data[8]; }

        double CPU_idle_time() const { return data[9]; }
        double& CPU_idle_time() { return data[9]; }

        double CPU_iowait_time() const { return data[10]; }
        double& CPU_iowait_time() { return data[10]; }

        double CPU_irq_time() const { return data[11]; }
        double& CPU_irq_time() { return data[11]; }

        double CPU_softirq_time() const { return data[12]; }
        double& CPU_softirq_time() { return data[12]; }

        double CPU_steal_time() const { return data[13]; }
        double& CPU_steal_time() { return data[13]; }

        double CPU_queue_length() const { return data[14]; }
        double& CPU_queue_length() { return data[14]; }

        //! Encode the state to a column vector.
        const arma::colvec& Encode() const { return data; }

        //! Dimension of the encoded state.
        static constexpr size_t dimension = 15;

       private:
        //! Locally-stored state data.
        arma::colvec data;
    }

    class Action {
       public:
        enum actions {
            allocate_core,
        }
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
    Serverless(const size_t maxSteps = 500, const double doneReward = 1.0)
        : maxSteps(maxSteps), doneReward(doneReward), StepsPerformed(0) {}

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
        StepsPerformed++;

        arma::colvec currentState = {
            state.TaskOnCore(),       state.TaskRespondTime(),
            state.TaskExecTime(),     state.TaskCPUtime(),
            state.TaskMemory(),       state.TaskPreemptions(),
            state.CPU_user_time(),    state.CPU_nice_time(),
            state.CPU_system_time(),  state.CPU_idle_time(),
            state.CPU_iowait_time(),  state.CPU_irq_time(),
            state.CPU_softirq_time(), state.CPU_steal_time(),
            state.CPU_queue_length()};

        double dest_core = action.action;
        nextState.TaskOnCore() = dest_core;
        // to do: add the logic to calculate the next state based on the current
        nextState.TaskRespondTime() = currentState.TaskRespondTime() + 1;
        nextState.TaskExecTime() = currentState.TaskExecTime() + 1;
        nextState.TaskCPUtime() = currentState.TaskCPUtime() + 1;
        nextState.TaskMemory() = currentState.TaskMemory() + 1;
        nextState.TaskPreemptions() = currentState.TaskPreemptions() + 1;
        nextState.CPU_user_time() = currentState.CPU_user_time() + 1;
        nextState.CPU_nice_time() = currentState.CPU_nice_time() + 1;
        nextState.CPU_system_time() = currentState.CPU_system_time() + 1;
        nextState.CPU_idle_time() = currentState.CPU_idle_time() + 1;
        nextState.CPU_iowait_time() = currentState.CPU_iowait_time() + 1;
        nextState.CPU_irq_time() = currentState.CPU_irq_time() + 1;
        nextState.CPU_softirq_time() = currentState.CPU_softirq_time() + 1;
        nextState.CPU_steal_time() = currentState.CPU_steal_time() + 1;
        nextState.CPU_queue_length() = currentState.CPU_queue_length() + 1;

        bool done = IsTerminal(nextState);

        if (done && maxSteps != 0 && StepsPerformed >= maxSteps) {
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
        stepPerformed = 0;
        // to do: add the logic to generate the initial state.
        return State();
    }

    /**
     * This function checks if the serverless has reached the terminal state.
     *
     * @param state The current state.
     * @return true if state is a terminal state, otherwise false.
     */
    bool IsTerminal(const State& state) const {
        // to do: add the logic to check if the state is terminal.
        return false;
    }

    // Get the number of steps performed.
    size_t StepsPerformed() const { return StepsPerformed; }

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
    size_t StepsPerformed;
};
}  // namespace mlpack

#endif