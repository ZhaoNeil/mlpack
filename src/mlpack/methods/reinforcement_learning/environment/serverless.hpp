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
        State() : data(dimension) { /* nothing to do here */ }

        /**
         * Construct a state instance from given data.
         *
         * @param data Data for the state.
         */
        // State(const arma::mat& data) : data(data) { /* Nothing to do here */
        // }
        State(const arma::rowvec& inputData) : data(inputData) {
            if (data.is_empty()) {
                throw std::runtime_error(
                    "State initialization error: data is empty.");
            }
        }

        // The historical response time (sum or average) of tasks completed on
        // this core.

        double StartedTasks() const {
            return data(0);
        }
        double& StartedTasks() {
            return data(0);
        }

        double UnstartedTasks() const {
            return data(1);
        }
        double& UnstartedTasks() {
            return data(1);
        }

        arma::rowvec& Data() { return data; }
        const arma::rowvec& Data() const { return data; }

        arma::colvec Encode() const { return data.t(); }

        static constexpr size_t dimension = 2;

        static constexpr size_t nMetrics = 2;

        static constexpr size_t nCores = 20;

       private:
        //! Locally-stored state data.
        arma::rowvec data;
    };
    
    /**
     * Construct a Serverless instance using the given constants.
     *
     * @param maxSteps The maximum number of steps allowed.
     * @param doneReward The reward recieved by the agent on success.
     *
     */
    Serverless(const size_t maxSteps, const double doneReward,
               size_t inputMetrics, size_t inputCores)
        : maxSteps(maxSteps),
          doneReward(doneReward),
          serverlessMetrics(inputMetrics),
          serverlessCores(inputCores),
          stepsPerformed(0) {}

    arma::rowvec GetLatestEnvironmentMetrics(const SharedData& sharedData) {
        sharedDataMutex_.Lock();
        arma::rowvec data = sharedData.getData();
        sharedDataMutex_.Unlock();
        return data;
    }

    // class Action {
    //    public:
    //     size_t action;

    //     Action() : action(0) {}
    //     Action(size_t id) : action(id) {}

    //     static constexpr size_t size = 20;
    // };

    class Action {
       public:
        static constexpr size_t kNumCores = State::nCores;  // 20
        static constexpr size_t kNumTimeBins = 3;
        static constexpr size_t kNumShuffleBins = 3;

        static constexpr size_t size =
            kNumCores * kNumTimeBins * kNumShuffleBins;

        // Flattened action id in [0, size-1]
        size_t action;

        Action() : action(0) {}
        explicit Action(size_t id) : action(id) {}

        // Helpers to pack/unpack a tuple <core, timeBin, shuffleBin>.
        static inline size_t Encode(size_t core, size_t timeBin,
                                    size_t shuffleBin) {
            return core + kNumCores * (timeBin + kNumTimeBins * shuffleBin);
        }
        static inline void Decode(size_t id, size_t& core, size_t& timeBin,
                                  size_t& shuffleBin) {
            core = id % kNumCores;
            id /= kNumCores;
            timeBin = id % kNumTimeBins;
            id /= kNumTimeBins;
            shuffleBin = id % kNumShuffleBins;  // in [0, kNumShuffleBins-1]
        }
        static inline constexpr uint16_t kTimeSliceMs[Action::kNumTimeBins] = {1000, 2000, 3000};
        static inline constexpr size_t kShuffleCounts[Action::kNumShuffleBins] = {2, 4, 6};
    };


    /**
     * Dynamics of the Serverless instance. Get reward and next state based on
     * current state and current action.
     *
     * @param state The current state.
     * @param action The current action.
     * @param nextState The next state.
     * @return reward,
     */
    // double Sample(const State& state, const Action& action, State& nextState) {
    //     stepsPerformed++;

    //     size_t dest_core = action.action;
    //     arma::rowvec latestdata = GetLatestEnvironmentMetrics(sharedData);
    //     nextState = State(latestdata);

    //     bool done = IsTerminal(nextState);
    //     if (done && maxSteps != 0 && stepsPerformed >= maxSteps) {
    //         return doneReward;
    //     }

    //     return - state.UnstartedTasks();
    // }

    double Sample(const State& state, const Action& action, State& nextState) {
        stepsPerformed++;
        // std::cout << "stepsPerformed= " << stepsPerformed << std::endl;

        size_t coreBin, timeBin, shuffleBin; //indices after decoding
        Action::Decode(action.action, coreBin, timeBin, shuffleBin);
        size_t dest_core = coreBin;
        double timeSlice = Action::kTimeSliceMs[timeBin];
        size_t shuffleCount = Action::kShuffleCounts[shuffleBin];

        arma::mat latestdata = GetLatestEnvironmentMetrics(sharedData);
        nextState = State(latestdata);

        bool done = IsTerminal(nextState);
        if (done && maxSteps != 0 && stepsPerformed >= maxSteps) {
            return doneReward;
        }

        return - state.StartedTasks();
    }

    /**
     * Dynamics of Serverless instance. Get reward based on current state
     * and action.
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
        // arma::mat currentdata = GetLatestEnvironmentMetrics(sharedData);
        // std::cout << "currentdata=" << currentdata << std::endl;
        arma::rowvec initialdata =
            arma::zeros<arma::rowvec>(State::nMetrics);

        return State(initialdata);
    }

    /**
     * This function checks if the serverless has reached the terminal
     * state.
     *
     * @param state The current state.
     * @return true if state is a terminal state, otherwise false.
     */
    bool IsTerminal(const State& state) const {
        if (maxSteps != 0 && stepsPerformed >= maxSteps) {
            std::cout << "Episode terminated due to the maximum number of steps"
                      << "being taken." << std::endl;
            return true;
        } else if (stepsPerformed > maxSteps - 1 &&
                   state.StartedTasks() == 0 && state.UnstartedTasks() == 0) {
            std::cout << "All tasks completed." << std::endl;
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

    // Dimension of the metrics
    size_t serverlessMetrics;

    // Dimension of the number of cores
    size_t serverlessCores;
};
}  // namespace mlpack

#endif