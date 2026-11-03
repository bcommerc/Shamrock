// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file EulerTimeDerivativeGasPassiveScalars.hpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @brief Per cell Euler time derivatives of the gas passive scalars
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_pscal_gas)                            \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_pscal_gas)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the Euler time derivatives of the gas passive scalars per cell
     *
     * Passive scalars are advected with the gas velocity, so \f$ \partial_t s = - \mathbf{v}
     * \cdot \nabla s \f$. Scalar \f$ j \f$ of cell \f$ i \f$ is at index \f$ i \, n_{\rm pscal} +
     * j \f$, while the gas velocity has one value per cell.
     */
    template<class Tvec>
    class NodeEulerTimeDerivativeGasPassiveScalars : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        u32 npscal_gas;

        public:
        NodeEulerTimeDerivativeGasPassiveScalars(u32 block_size, u32 npscal_gas)
            : block_size(block_size), npscal_gas(npscal_gas) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "EulerTimeDerivativeGasPassiveScalars";
        };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
