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
 * @file state_pscal_gas.hpp
 * @author Leodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 * @brief
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
namespace shammath {

    template<class Tvec_>
    struct ConsStatePscal {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rhopscal{};

        const ConsStatePscal &operator+=(const ConsStatePscal &);
        const ConsStatePscal &operator-=(const ConsStatePscal &);
        const ConsStatePscal &operator*=(const Tscal);
    };

    template<class Tvec_>
    struct PrimStatePscal {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal pscal{};
    };

    template<class Tvec>
    const ConsStatePscal<Tvec> &ConsStatePscal<Tvec>::operator+=(const ConsStatePscal<Tvec> &cst) {
        rhopscal += cst.rhopscal;
        return *this;
    }

    template<class Tvec>
    const ConsStatePscal<Tvec>
    operator+(const ConsStatePscal<Tvec> &lhs, const ConsStatePscal<Tvec> &rhs) {
        return ConsStatePscal<Tvec>(lhs) += rhs;
    }

    template<class Tvec>
    const ConsStatePscal<Tvec> &ConsStatePscal<Tvec>::operator-=(const ConsStatePscal<Tvec> &cst) {
        rhopscal -= cst.rhopscal;
        return *this;
    }

    template<class Tvec>
    const ConsStatePscal<Tvec>
    operator-(const ConsStatePscal<Tvec> &lhs, const ConsStatePscal<Tvec> &rhs) {
        return ConsStatePscal<Tvec>(lhs) -= rhs;
    }

    template<class Tvec>
    const ConsStatePscal<Tvec> &
    ConsStatePscal<Tvec>::operator*=(const typename ConsStatePscal<Tvec>::Tscal factor) {
        rhopscal *= factor;
        return *this;
    }

    template<class Tvec>
    const ConsStatePscal<Tvec>
    operator*(const typename ConsStatePscal<Tvec>::Tscal factor, const ConsStatePscal<Tvec> &rhs) {
        return ConsStatePscal<Tvec>(rhs) *= factor;
    }

    template<class Tvec>
    const ConsStatePscal<Tvec>
    operator*(const ConsStatePscal<Tvec> &lhs, const typename ConsStatePscal<Tvec>::Tscal factor) {
        return ConsStatePscal<Tvec>(lhs) *= factor;
    }

    template<class Tvec_>
    struct FluxesPscal {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<ConsStatePscal<Tvec>, 3> F;
    };

    template<class Tvec>
    inline constexpr ConsStatePscal<Tvec> prim_to_cons_pscal(
        const PrimStatePscal<Tvec> primpscal, typename ConsStatePscal<Tvec>::Tscal rhogas) {
        ConsStatePscal<Tvec> cons;

        cons.rhopscal = rhogas * primpscal.pscal;

        return cons;
    }

    template<class Tvec>
    inline constexpr PrimStatePscal<Tvec> cons_to_prim_pscal(
        const ConsStatePscal<Tvec> cons, typename ConsStatePscal<Tvec>::Tscal rhogas) {
        PrimStatePscal<Tvec> prim;

        prim.pscal = cons.rhopscal / rhogas;

        return prim;
    }

    template<class Tvec>
    inline constexpr ConsStatePscal<Tvec> hydro_flux_x(
        const ConsStatePscal<Tvec> cons, typename ConsStatePscal<Tvec>::Tscal rhogas, Tvec vgas) {
        ConsStatePscal<Tvec> flux;

        const PrimStatePscal<Tvec> prim = cons_to_prim_pscal(cons, rhogas);

        flux.rhopscal = cons.rhopscal * vgas[0];

        return flux;
    }

    template<class Tvec>
    inline constexpr ConsStatePscal<Tvec> hydro_flux_y(
        const ConsStatePscal<Tvec> cons, typename ConsStatePscal<Tvec>::Tscal rhogas, Tvec vgas) {
        ConsStatePscal<Tvec> flux;

        const PrimStatePscal<Tvec> prim = cons_to_prim_pscal(cons, rhogas);

        flux.rhopscal = cons.rhopscal * vgas[1];

        return flux;
    }

    template<class Tvec>
    inline constexpr ConsStatePscal<Tvec> hydro_flux_z(
        const ConsStatePscal<Tvec> cons, typename ConsStatePscal<Tvec>::Tscal rhogas, Tvec vgas) {
        ConsStatePscal<Tvec> flux;

        const PrimStatePscal<Tvec> prim = cons_to_prim_pscal(cons, rhogas);

        flux.rhopscal = cons.rhopscal * vgas[2];

        return flux;
    }

} // namespace shammath
