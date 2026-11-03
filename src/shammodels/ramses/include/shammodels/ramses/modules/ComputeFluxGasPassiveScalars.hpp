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
 * @file ComputeFluxGasPassiveScalars.hpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @brief Upwind fluxes of the gas passive scalars from the gas mass flux
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/solvegraph/NeighGraphLinkFieldEdge.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <array>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(LinkFieldFlux, flux_rho_face_xp)                                                          \
    X_RO(LinkFieldFlux, flux_rho_face_xm)                                                          \
    X_RO(LinkFieldFlux, flux_rho_face_yp)                                                          \
    X_RO(LinkFieldFlux, flux_rho_face_ym)                                                          \
    X_RO(LinkFieldFlux, flux_rho_face_zp)                                                          \
    X_RO(LinkFieldFlux, flux_rho_face_zm)                                                          \
    X_RO(LinkFieldScal, pscal_gas_face_xp)                                                         \
    X_RO(LinkFieldScal, pscal_gas_face_xm)                                                         \
    X_RO(LinkFieldScal, pscal_gas_face_yp)                                                         \
    X_RO(LinkFieldScal, pscal_gas_face_ym)                                                         \
    X_RO(LinkFieldScal, pscal_gas_face_zp)                                                         \
    X_RO(LinkFieldScal, pscal_gas_face_zm)                                                         \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(LinkFieldFlux, flux_pscal_gas_face_xp)                                                    \
    X_RW(LinkFieldFlux, flux_pscal_gas_face_xm)                                                    \
    X_RW(LinkFieldFlux, flux_pscal_gas_face_yp)                                                    \
    X_RW(LinkFieldFlux, flux_pscal_gas_face_ym)                                                    \
    X_RW(LinkFieldFlux, flux_pscal_gas_face_zp)                                                    \
    X_RW(LinkFieldFlux, flux_pscal_gas_face_zm)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the fluxes of the gas passive scalars on every face
     *
     * The scalars are carried by the gas, so their flux is the gas mass flux times the upwind
     * scalar value: \f$ F_{s} = F_\rho \, s_{\rm upwind} \f$, with \f$ s_{\rm upwind} \f$ taken
     * on the side the gas comes from. This keeps \f$ s \f$ exactly constant when it is uniform.
     *
     * The gas mass flux is along the outward normal of the face, from the cell to its neighbour,
     * so \f$ F_\rho \ge 0 \f$ means the gas leaves the cell. Scalar \f$ k \f$ of link \f$ l \f$ is
     * at index \f$ l \, n_{\rm pscal} + k \f$.
     */
    template<class Tvec, class TgridVec>
    class NodeComputeFluxGasPassiveScalars : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Aliases without commas, the edge list macros cannot carry template argument lists
        using AMRGraphEdge  = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        using LinkFieldFlux = solvergraph::NeighGraphLinkFieldEdge<Tscal>;
        using LinkFieldScal = solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>;

        u32 npscal_gas;

        public:
        NodeComputeFluxGasPassiveScalars(u32 npscal_gas) : npscal_gas(npscal_gas) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "ComputeFluxGasPassiveScalars";
        }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
