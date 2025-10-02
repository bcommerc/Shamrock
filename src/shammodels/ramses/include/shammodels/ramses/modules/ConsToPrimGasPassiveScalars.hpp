// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ConsToPrimGasPassiveScalars.hpp
 * @author Benoît Commerçon (benoit.commercon@ens-lyon.fr)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2025-06-26
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::basegodunov::modules {
    template<class Tvec>
    class NodeConsToPrimGasPassiveScalars : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        u32 npscal_gas;

        public:
        NodeConsToPrimGasPassiveScalars(u32 block_size, u32 npscal_gas)
            : block_size(block_size), npscal_gas(npscal_gas) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldRefs<Tscal> &spans_rho_pscal;
            const shamrock::solvergraph::IFieldRefs<Tscal> &spans_rho_gas;
            shamrock::solvergraph::IFieldRefs<Tscal> &spans_pscal;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_rho_pscal,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_rho_gas,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_pscal) {
            __internal_set_ro_edges({sizes, spans_rho_pscal, spans_rho_gas});
            __internal_set_rw_edges({spans_pscal});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ConsToPrimGasPassiveScalars"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
