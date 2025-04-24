/**
 * @file methods/reinforcement_learning/q_networks/tabular_q.hpp
 * @author 
 *
 * A minimal tabular Q network that satisfies the
 * `Forward() / Backward() / Parameters() / Reset()` interface expected by
 * `mlpack::rl::QLearning`.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_TABULAR_Q_HPP
#define MLPACK_METHODS_RL_TABULAR_Q_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename MatType = arma::mat>
class TabularQ
{
 public:
  /** Default-construct an *empty* table (needed for the target network). */
  TabularQ() : states(0), actions(0), q() {}

  /** Construct an |S| × |A| zero-initialised table. */
  TabularQ(size_t states, size_t actions)
      : states(states), actions(actions),
        q(states, actions, arma::fill::zeros) {}

  /* --- Inference ------------------------------------------------------- */

  //! Batch prediction used by QLearning::SelectAction().
  void Predict(const MatType& encodedState, MatType& actionValue)
  { Forward(encodedState, actionValue); }

  //! Forward pass (same signature SimpleDQN exposes).
  void Forward(const MatType& encodedState, MatType& out)
  {
    const size_t batch = encodedState.n_cols;
    out.set_size(actions, batch);
    for (size_t col = 0; col < batch; ++col)
    {
      const size_t s = (size_t) encodedState(0, col);
      out.col(col) = q.row(s).t();
    }
  }

  /* --- Back-prop (the TD error is supplied as 'gradient') -------------- */

  void Backward(const MatType& encodedState,
                const MatType& /* target */,
                MatType& gradient)
  {
    const size_t batch = encodedState.n_cols;
    for (size_t col = 0; col < batch; ++col)
    {
      const size_t s = (size_t) encodedState(0, col);
      q.row(s) += gradient.col(col).t();   // plain SGD on that row
    }
  }

  /* --- Plumbing required by mlpack ------------------------------------ */

  //! Access parameters so the optimiser can see them.
  MatType&       Parameters()       { return q; }
  const MatType& Parameters() const { return q; }

  //! Re-initialise when the state dimensionality changes.
  void Reset(size_t newStates = 0)
  {
    if (newStates != 0)
    {
      states = newStates;
      q.set_size(states, actions);
      q.zeros();
    }
  }

  void ResetNoise() {}                 // nothing noisy in a table

 private:
  size_t states, actions;
  MatType q;
};

} // namespace mlpack

#endif // MLPACK_METHODS_RL_TABULAR_Q_HPP
