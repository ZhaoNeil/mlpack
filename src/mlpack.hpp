/**
 * @file mlpack.hpp
 *
 * Include all of mlpack!  When this file is included, all components of mlpack
 * are available.
 *
 * Note that by default, serialization for ANN layers is not enabled, since this
 * will cause the build time to be very long.  If you plan to serialize a neural
 * network, simply include mlpack like this:
 *
 * ```
 * #define MLPACK_ENABLE_ANN_SERIALIZATION
 * #include <mlpack.hpp>
 * ```
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_HPP
#define MLPACK_HPP

#include <armadillo>
#include "absl/synchronization/mutex.h"

#include <mutex>

class SharedData {
private:
    arma::mat data;
    size_t nMetrics;
    size_t nCores;

public:
    SharedData(size_t metrics, size_t cores)
        : nMetrics(metrics),
          nCores(cores),
          data(nMetrics, nCores, arma::fill::zeros) {}

    arma::mat getData() const {
        return data;
    }

    void setData(const arma::mat& newData) {
        if (newData.n_rows == nMetrics && newData.n_cols == nCores) {
            data = newData;
        } else {
            // Potentially handle mismatch here, or just assign as you do now.
            data = newData;  // Possibly resizing here if that’s appropriate.
        }
    }

    void setValue(size_t metricIndex, size_t coreIndex, double value) {
        if (metricIndex < nMetrics && coreIndex < nCores) {
            data(metricIndex, coreIndex) = value;
        }
    }

    double getValue(size_t metricIndex, size_t coreIndex) const {
        if (metricIndex < nMetrics && coreIndex < nCores) {
            return data(metricIndex, coreIndex);
        }
        return 0.0;
    }
};

extern SharedData sharedData;
extern absl::Mutex sharedDataMutex_;

// Include all of the core library components.
#include "mlpack/base.hpp"
#include "mlpack/core.hpp"
#include "mlpack/prereqs.hpp"

// Now include all of the methods.
#include "mlpack/methods/adaboost.hpp"
#include "mlpack/methods/amf.hpp"
#include "mlpack/methods/ann.hpp"
#include "mlpack/methods/approx_kfn.hpp"
#include "mlpack/methods/bayesian_linear_regression.hpp"
#include "mlpack/methods/bias_svd.hpp"
#include "mlpack/methods/block_krylov_svd.hpp"
#include "mlpack/methods/cf.hpp"
#include "mlpack/methods/dbscan.hpp"
#include "mlpack/methods/decision_tree.hpp"
#include "mlpack/methods/det.hpp"
#include "mlpack/methods/emst.hpp"
#include "mlpack/methods/fastmks.hpp"
#include "mlpack/methods/gmm.hpp"
#include "mlpack/methods/hmm.hpp"
#include "mlpack/methods/hoeffding_trees.hpp"
#include "mlpack/methods/kde.hpp"
#include "mlpack/methods/kernel_pca.hpp"
#include "mlpack/methods/kmeans.hpp"
#include "mlpack/methods/lars.hpp"
#include "mlpack/methods/linear_regression.hpp"
#include "mlpack/methods/linear_svm.hpp"
#include "mlpack/methods/lmnn.hpp"
#include "mlpack/methods/local_coordinate_coding.hpp"
#include "mlpack/methods/logistic_regression.hpp"
#include "mlpack/methods/lsh.hpp"
#include "mlpack/methods/matrix_completion.hpp"
#include "mlpack/methods/mean_shift.hpp"
#include "mlpack/methods/naive_bayes.hpp"
#include "mlpack/methods/nca.hpp"
#include "mlpack/methods/neighbor_search.hpp"
#include "mlpack/methods/nmf.hpp"
#include "mlpack/methods/nystroem_method.hpp"
#include "mlpack/methods/pca.hpp"
#include "mlpack/methods/perceptron.hpp"
#include "mlpack/methods/preprocess.hpp"
#include "mlpack/methods/quic_svd.hpp"
#include "mlpack/methods/radical.hpp"
#include "mlpack/methods/random_forest.hpp"
#include "mlpack/methods/randomized_svd.hpp"
#include "mlpack/methods/range_search.hpp"
#include "mlpack/methods/rann.hpp"
#include "mlpack/methods/regularized_svd.hpp"
#include "mlpack/methods/reinforcement_learning.hpp"
#include "mlpack/methods/softmax_regression.hpp"
#include "mlpack/methods/sparse_autoencoder.hpp"
#include "mlpack/methods/sparse_coding.hpp"
#include "mlpack/methods/svdplusplus.hpp"

// Include reverse compatibility.
#include "mlpack/namespace_compat.hpp"

#endif
