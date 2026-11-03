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
 * @file SumFluxGasPassiveScalars.hpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @brief Sum the fluxes into the time derivative fields for the gas passive scalars
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/solvegraph/NeighGraphLinkFieldEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    /* number of blocks     */                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, block_counts)                                        \
                                                                                                   \
    /* cell neighbor graph */                                                                      \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
                                                                                                   \
    /* block cell sizes */                                                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    /* block low corner position */                                                                \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_cell0block_aabb_lower)                     \
                                                                                                   \
    /* fluxes */                                                                                   \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_pscal_gas_face_xp)                      \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_pscal_gas_face_xm)                      \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_pscal_gas_face_yp)                      \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_pscal_gas_face_ym)                      \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_pscal_gas_face_zp)                      \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_pscal_gas_face_zm)                      \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dtrho_gas_pscal) /* d(rho s)/dt */

namespace shammodels::basegodunov::modules {

    /**
     * @brief Sum the passive scalar fluxes of the faces of each cell into
     * \f$ \partial_t (\rho s) \f$
     *
     * The faces are visited in the same fixed order as for the gas and the dust, so the sum is
     * deterministic.
     */
    template<class Tvec, class TgridVec>
    class NodeSumFluxGasPassiveScalars : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = shammodels::amr::AMRBlock<Tvec, TgridVec, 1>;

        u32 block_size;
        Tscal dxfact;
        u32 npscal_gas;

        public:
        NodeSumFluxGasPassiveScalars(u32 block_size, Tscal dxfact, u32 npscal_gas)
            : block_size(block_size), dxfact(dxfact), npscal_gas(npscal_gas) {}

        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SumFluxGasPassiveScalars"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
