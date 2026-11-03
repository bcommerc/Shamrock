// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeFluxGasPassiveScalars.cpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ComputeFluxGasPassiveScalars.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec, class TgridVec>
    void compute_flux_pscal_gas_dir(
        u32 npscal_gas,
        shammodels::basegodunov::modules::Direction dir,
        const shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>
            &cell_neigh_graph,
        const shammodels::basegodunov::solvergraph::NeighGraphLinkFieldEdge<
            shambase::VecComponent<Tvec>> &flux_rho_face,
        const shammodels::basegodunov::solvergraph::NeighGraphLinkFieldEdge<
            std::array<shambase::VecComponent<Tvec>, 2>> &pscal_gas_face,
        shammodels::basegodunov::solvergraph::NeighGraphLinkFieldEdge<shambase::VecComponent<Tvec>>
            &flux_pscal_gas_face) {

        using Tscal = shambase::VecComponent<Tvec>;

        auto graphs_dir = cell_neigh_graph.get_refs_dir(dir);

        flux_rho_face.check_size(graphs_dir);
        pscal_gas_face.check_size(graphs_dir);

        flux_pscal_gas_face.resize_according_to(graphs_dir);

        shambase::DistributedData<u32> counts_dir
            = graphs_dir.template map<u32>([&](u64 id, auto &graph) {
                  return graph.get().link_count * npscal_gas;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{flux_rho_face.link_fields, pscal_gas_face.link_fields},
            sham::DDMultiRef{flux_pscal_gas_face.link_fields},
            counts_dir,
            [npscal_gas](
                u32 idvar,
                const Tscal *__restrict flux_rho,
                const std::array<Tscal, 2> *__restrict pscal_face,
                Tscal *__restrict flux_pscal) {
                u32 link_id = idvar / npscal_gas;

                Tscal f_rho                 = flux_rho[link_id];
                std::array<Tscal, 2> s_face = pscal_face[idvar];

                // upwind: the cell side when the gas leaves the cell, the neighbour side otherwise
                Tscal s_upwind = (f_rho >= 0) ? s_face[0] : s_face[1];

                flux_pscal[idvar] = f_rho * s_upwind;
            });
    }

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeComputeFluxGasPassiveScalars<Tvec, TgridVec>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        auto edges = get_edges();

        compute_flux_pscal_gas_dir<Tvec, TgridVec>(
            npscal_gas,
            Direction::xp,
            edges.cell_neigh_graph,
            edges.flux_rho_face_xp,
            edges.pscal_gas_face_xp,
            edges.flux_pscal_gas_face_xp);
        compute_flux_pscal_gas_dir<Tvec, TgridVec>(
            npscal_gas,
            Direction::xm,
            edges.cell_neigh_graph,
            edges.flux_rho_face_xm,
            edges.pscal_gas_face_xm,
            edges.flux_pscal_gas_face_xm);
        compute_flux_pscal_gas_dir<Tvec, TgridVec>(
            npscal_gas,
            Direction::yp,
            edges.cell_neigh_graph,
            edges.flux_rho_face_yp,
            edges.pscal_gas_face_yp,
            edges.flux_pscal_gas_face_yp);
        compute_flux_pscal_gas_dir<Tvec, TgridVec>(
            npscal_gas,
            Direction::ym,
            edges.cell_neigh_graph,
            edges.flux_rho_face_ym,
            edges.pscal_gas_face_ym,
            edges.flux_pscal_gas_face_ym);
        compute_flux_pscal_gas_dir<Tvec, TgridVec>(
            npscal_gas,
            Direction::zp,
            edges.cell_neigh_graph,
            edges.flux_rho_face_zp,
            edges.pscal_gas_face_zp,
            edges.flux_pscal_gas_face_zp);
        compute_flux_pscal_gas_dir<Tvec, TgridVec>(
            npscal_gas,
            Direction::zm,
            edges.cell_neigh_graph,
            edges.flux_rho_face_zm,
            edges.pscal_gas_face_zm,
            edges.flux_pscal_gas_face_zm);
    }

    template<class Tvec, class TgridVec>
    std::string NodeComputeFluxGasPassiveScalars<Tvec, TgridVec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Upwind fluxes of the gas passive scalars (shown for the $+x$ faces)

            \begin{align}
            {flux_pscal_gas_face_xp}_{l,k} &= {flux_rho_face_xp}_l \times \begin{cases}
                {pscal_gas_face_xp}_{l,k,0} & {flux_rho_face_xp}_l \geq 0 \\
                {pscal_gas_face_xp}_{l,k,1} & {flux_rho_face_xp}_l < 0
            \end{cases} \\
            k &\in [0,n_{\rm pscal}) \\
            n_{\rm pscal} & = {npscal_gas}
            \end{align}
        )tex";

        replace_edges_tex_symbols(tex);

        shambase::replace_all(tex, "{npscal_gas}", sham::format("{}", npscal_gas));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeFluxGasPassiveScalars<f64_3, i64_3>;
