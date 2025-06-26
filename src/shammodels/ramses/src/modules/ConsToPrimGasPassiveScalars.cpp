// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrimGasPassiveScalars.cpp
 * @author Benoît Commerçon (benoit.commercon@ens-lyon.fr)
 * @brief
 * @date 2025-06-26
 */

#include "shammodels/ramses/modules/ConsToPrimGasPassiveScalars.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelConsToPrimGasPassiveScalars {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_pscal,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_pscal,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size,
            u32 npscal_gas) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_rho_pscal},
                sham::DDMultiRef{spans_pscal},
                cell_counts,
                [npscal_gas](u32 i, const Tscal *__restrict rhopscal, Tscal *__restrict pscal) {
                    auto pscal_conststate = shammath::ConsState<Tvec>{rhopscal[i]};

                    auto pscal_prim_state = shammath::pscal_cons_to_prim(pscal_conststate);

                    pscal[i] = prim_state.pscal;
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeConsToPrimGas<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);

        edges.spans_vel.ensure_sizes(edges.sizes.indexes);
        edges.spans_P.ensure_sizes(edges.sizes.indexes);

        KernelConsToPrimGas<Tvec>::kernel(
            edges.spans_rho.get_spans(),
            edges.spans_rhov.get_spans(),
            edges.spans_rhoe.get_spans(),
            edges.spans_vel.get_spans(),
            edges.spans_P.get_spans(),
            edges.sizes.indexes,
            block_size,
            gamma);
    }

    template<class Tvec>
    std::string NodeConsToPrimGas<Tvec>::_impl_get_tex() {

        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto rho         = get_ro_edge_base(1).get_tex_symbol();
        auto rhov        = get_ro_edge_base(2).get_tex_symbol();
        auto rhoe        = get_ro_edge_base(3).get_tex_symbol();
        auto vel         = get_rw_edge_base(0).get_tex_symbol();
        auto P           = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Conservative to primitive variable (gas)

            \begin{align}
            {vel}_i &= \frac{ {rhov}_i }{ {rho}_i } \\
            {P}_i &= (\gamma - 1) \left( {rhoe}_i - \frac{ {rhov}_i^2 }{ 2 {rho}_i } \right) \\
            i &\in [0,{block_count} * N_{\rm cell/block}) \\
            \gamma &= {gamma} \\
            N_{\rm cell/block} & = {block_size}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{vel}", vel);
        shambase::replace_all(tex, "{P}", P);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{rhov}", rhov);
        shambase::replace_all(tex, "{rhoe}", rhoe);
        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{gamma}", shambase::format("{}", gamma));
        shambase::replace_all(tex, "{block_size}", shambase::format("{}", block_size));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeConsToPrimGas<f64_3>;
