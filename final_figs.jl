# loading modules
using PyPlot, Printf, LinearAlgebra, FFTW, FileIO, JLD2

using CUDA

using Setfield

# include("../../LinStab/mjcl_stab.jl")
# using .LinStab

using FourierFlows: CPU, TwoDGrid, plan_flows_rfft

using GeophysicalFlows

using Glob

using Statistics

using Parameters

using StaticArrays, KernelAbstractions

run_file_dir = "/scratch/cimes/ml1994/QG/julia/QG_topo/"
include(run_file_dir * "mjcl_gfjl_fcns.jl")

matplotlib[:rc]("text", usetex= "true")
matplotlib[:rc]("text.latex", preamble= "\\usepackage{amsmath,amsthm}")

using PlotUtils


dev = GeophysicalFlows.CPU()

Ny = Nx = n = 1024                # 2D resolution = n²


###################################################################################
## Geometry and background fields
###################################################################################

Nz = nlayers = 2                 # number of layers
f0 = f₀= 8.3e-5            # Coriolis param [s^-1]
g = 9.81                    # gravity
H = [2000., 2000.]     # the rest depths of each layer;

Us = [0.01, 0.017634058790440166, 0.024]
rhos = 1025.75 - 0.5775

kt = 0.

rho0 = 1025.        # Boussinesq reference density

# always zero for now, model can't even take this in as a parameter...
V = zeros(nlayers)


# running model

rho = ρ = [0., 1025.75]
rho[1] = ρ[1] = rhos[1]

strat_str = "uni_strat"
shear_str = "uni_shear"

# change U here
U = [0.,0.]
U[1] = Us[1]

gprime = (g/rho0)*(ρ[2] - ρ[1])

Ld = sqrt(gprime * sum(H)) / 2 / f0

L = Lx = Ly = 25*2*pi*Ld   # domain size [m]

###################################################################################
## Planetary PV gradient
###################################################################################
betas = βs = collect(range(-1.5,1.5,13)) * (U[1]/2) * Ld^-2
β = 0. # 1e-11              # the y-gradient of planetary PV; beta = β = 0. * (U[1]/2) * Ld^-2

###################################################################################
## Topography
###################################################################################

# set topography w/ current options:
topo_type = "y_slope"

# Defining a bulk slope parameter...should probably change this!

# setting slope(s)
S32 = f0 * U[1] / gprime

dyetab_over_S32 = collect(range(-1.5,1.5,13))
h0s = dyetab_over_S32 .* S32
# h0s = h0s[7:end]

# tuple of (bottom slope, h_rms)
h0 = (h0s[1], 0.)    #

# central wavenumber for rough topo
kt = 0.

# defining GFJL topo params
interim_params = mod_params(topo_type=topo_type, h0=h0, kt=kt)

topographic_pv_gradient, eta = define_topo(interim_params)

###################################################################################
# Dissipation
###################################################################################

# biharmonic viscosity
nν = 4
νstar = 0. # 10^-13
ν = νstar * ((U[1]-U[end])/2) * (Lx/2/pi)^7

dyn_nu = false

# Linear bottom drag
μstars = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
mus = @. μstars * ((U[1]-U[end]) / 2) / Ld

# Quadratic bottom drag
κstars = [0.]
kappas = [0.] # @. κstars / Ld

###################################################################################
## Time stepping and sample rate
###################################################################################

if ν==0.
    stepper = "FilteredETDRK4"
    # af = 0.
    # global filt_order=4
    # global innerK = 2/3
else
    stepper = "ETDRK4"
    # af = 1/3
    # global filt_order=4
    # innerK = 2/3
end

# should I set dt via CFL condition instead?
dt = 600.

###################################################################################
## Details of saving data
###################################################################################
global diags = diag_bools(two_layer_kspace_modal_nrg_budget_bool=true);

# where to save model output; frequency of output
# data_dir = "/scratch/cimes/ml1994/QG/julia/data/sloping_LD_two_layer_data/"

data_dir = "/scratch/cimes/ml1994/QG/julia/data/JFM_May2025_data/"

f_s = 5. * (Ld / ((U[1]-U[end])/2)) #  N eddy turnover periods

# sampling period in time steps (set above)
nsubs = round(Int64, f_s/dt)

###################################################################################
## Model run parameters
###################################################################################

# steady-state  parameters
ss_yr_max = ceil(100 * (Ld / ((U[1]-U[end])/2)) / 3600 / 24 / 365.25)  # number of eddy periods converted to years for model run
yr_increment = 1.0 # how often to write to a save file

restart_bool = false

pre_buoy_restart_file = true
data_dir_pre_buoy = "/scratch/cimes/ml1994/QG/julia/data/two_layer_drag_study/"

###################################################################################
## Define params struct
###################################################################################
global model_params = mod_params(
data_dir = data_dir,
Nz = Nz, Nx = Nx, Ny = Ny, Lx = Lx, Ly = Ly, Ld = Ld,
H = H,
rho0 = rho0, rho = rho, strat_str = strat_str,
shear_str = shear_str, U = U,
μ = mus[1], κ = kappas[1], nν = nν, ν = ν, dyn_nu = dyn_nu,
eta = eta, topographic_pv_gradient = topographic_pv_gradient, topo_type = topo_type, h0 = h0,
β = β,
dt = dt,
stepper = stepper,
dev = dev,
restart_bool = restart_bool,
pre_buoy_restart_file = pre_buoy_restart_file,
data_dir_pre_buoy = data_dir_pre_buoy,
ss_yr_max = ss_yr_max,
yr_increment = yr_increment,
nsubs = nsubs)

###################################################################################
## Initialize model
###################################################################################

prob, prob_filt = initialize_model(model_params);

# prob = set_initial_conditions(prob, prob_filt, model_params)

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid;


dev = grid.device
T = eltype(grid)
A = device_array(dev)

rfftplan = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, 1), [1, 2]; flags=FFTW.MEASURE);

function redef_mu_kappa(model_params, mu, kappa)
    @unpack_mod_params model_params

    mp_out = mod_params(
    data_dir = data_dir,
    Nz = Nz, Nx = Nx, Ny = Ny, Lx = Lx, Ly = Ly, Ld = Ld,
    H = H,
    rho0 = rho0, rho = rho, strat_str = strat_str,
    shear_str = shear_str, U = U,
    μ = mu , κ = kappa , nν = nν, ν = ν, dyn_nu=dyn_nu,
    eta = eta, topographic_pv_gradient = topographic_pv_gradient, topo_type = topo_type, h0 = h0,
    β = β,
    dt = dt,
    stepper = stepper,
    dev = dev,
    restart_bool = restart_bool,
    ss_yr_max = ss_yr_max,
    yr_increment = yr_increment,
    nsubs = nsubs);

    return mp_out
end


using PlotUtils

cm = cgrad(:coolwarm);
clrs = [cm[i] for i in range(0,1,100)]

hc_range = collect(range(h0s[1],h0s[end],100))

function h_color(h_in,hc_range,colors)
    c_ind = argmin(abs.(h_in .- (hc_range)))

    color_out = [red(clrs[c_ind]),green(clrs[c_ind]),blue(clrs[c_ind])]

    if h_in==0
        color_out = [0.,0.,0.]
    end
    return color_out
end

####################################################################
##
####################################################################

function plot_info_box(ax, var, x, y, plus_mode, fsize)
    textstr = var * "\n" * L"+ \rightarrow" * plus_mode

    # these are matplotlib.patch.Patch properties
    props = Dict("boxstyle"=>"round", "facecolor"=>"wheat", "alpha"=>0.5)

    # place a text box in upper left in axes coords
    ax.text(x, y, textstr, transform=ax.transAxes, fontsize=fsize,
            ha="left", va="top", bbox=props)

    return nothing
end


function plot_zetaBT(a, prob, ax)
    ψ = a["jld_data"]["psi_yrs_end"]

    grid = prob.grid

    # assigning basic variables
    ψBC = 0.5 * (ψ[:,:,1] .- ψ[:,:,2])
    ψBT = 0.5 * (ψ[:,:,1] .+ ψ[:,:,2])

    ψBCh = deepcopy(vars.uh[:,:,1])
    ψBTh = deepcopy(vars.uh[:,:,1])

    mul2D!(ψBCh, rfftplan, ψBC)
    mul2D!(ψBTh, rfftplan, ψBT)

    ζBCh = - grid.Krsq .* ψBCh
    ζBTh = -im .* grid.l .* ψBTh  # - grid.Krsq .* ψBTh

    ζBC = deepcopy(vars.u[:,:,1])
    ζBT = deepcopy(vars.u[:,:,1])

    ldiv2D!(ζBC, rfftplan, ζBCh)
    ldiv2D!(ζBT, rfftplan, ζBTh)

    ##
    ∂xψBTh = im * grid.kr .* ψBTh
    ∂yψBTh = im * grid.l .* ψBTh

    ∂xψBT = deepcopy(vars.u[:,:,1])
    ∂yψBT = deepcopy(vars.u[:,:,1])

    ldiv2D!(∂xψBT, rfftplan, im * grid.kr .* ψBTh)
    ldiv2D!(∂yψBT, rfftplan, im * grid.l .* ψBTh)

    ∂xψBCh = im * grid.kr .* ψBCh
    ∂yψBCh = im * grid.l .* ψBCh

    ∂xψBC = deepcopy(vars.u[:,:,1])
    ∂yψBC = deepcopy(vars.u[:,:,1])

    ldiv2D!(∂xψBC, rfftplan, im * grid.kr .* ψBCh)
    ldiv2D!(∂yψBC, rfftplan, im * grid.l .* ψBCh)

    EKE_xy = @. ∂xψBT^2 + ∂yψBT^2
    ##

    norm = (a["jld_data"]["two_layer_vBT_scale"])^-2
    lim = maximum(abs.(EKE_xy .* norm))/2

    x = collect(range(0, Lx, 1025)) ./ (Ld * 2 * pi)
    y = collect(range(0, Lx, 1025)) ./ (Ld * 2 * pi)

    ax.pcolormesh(x, y, (EKE_xy .* norm)', cmap=PyPlot.cm.bwr, vmin=-lim, vmax=lim, rasterized=true)
end



####################################################################
##
####################################################################


# tab20_range = collect(range(0,1,20))

function tab20_color(c_ind)

    cm2 = cgrad(:tab20);
    clrs2 = [cm2[i] for i in range(0,1,20)]

    color_out = [red(clrs2[c_ind]),green(clrs2[c_ind]),blue(clrs2[c_ind])]

    return color_out
end



function plot_budget(ax1, ax2, ax3, i, j, k, legend_bool)

    h0 = (h0s[k], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[i]

    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

    beta = betas[7]

    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

    a = load(data_dir*jld_name(model_params,71.95))


    norm = a["jld_data"]["two_layer_vBT_scale"]^-3 * Ld^-1

    ##############################################################
    # EAPE budget
    ##############################################################
    # trans_resid = -(a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .+ 2 .* a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13] )
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .* norm, label=L"\widehat{C}_\mathrm{BC}", color=tab20_color(1), linestyle="solid", linewidth=lwax)
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .* norm , label=L"\widehat{N}^\mathrm{PE}_\mathrm{BC}", color=tab20_color(19), linestyle="dashed", linewidth=lwax)
    # ax1.plot(grid.kr*Ld, 2 * a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"-2 \, D_\mathrm{BC}")
    ax1.plot(grid.kr*Ld,a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"\widehat{T}^\mathrm{D}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)

    resid_EAPE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4]

    ax1.plot(grid.kr*Ld,resid_EAPE .* norm, color="black", alpha=1.0, linewidth=0.5)

    if i==2
        # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # axins = ax1.inset_axes(ax[1], width="30%", height="40%")
        axins = ax1.inset_axes([0.125, 0.6, 1-0.125, 0.2])

        axins.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .* norm, label=L"\widehat{C}_\mathrm{BC}", color=tab20_color(1), linestyle="solid", linewidth=lwax)
        axins.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .* norm , label=L"\widehat{N}^\mathrm{PE}_\mathrm{BC}", color=tab20_color(19), linestyle="dashed", linewidth=lwax)
        # axins.plot(grid.kr*Ld, 2 * a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"-2 \, D_\mathrm{BC}")
        axins.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"\widehat{T}^\mathrm{D}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)

        axins.plot(grid.kr*Ld,resid_EAPE .* norm, color="black", alpha=1.0, linewidth=0.5)

        axins.set_xlim(0.125, 1.5)
        axins.set_ylim(-0.005, 0.005)

        axins.set_yticklabels([])
        axins.set_xticklabels([])

    end


    ##############################################################
    # BC EKE budget
    ##############################################################
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"-\widehat{T}^\mathrm{D}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"-\widehat{T}_\mathrm{flat}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    # ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"D_\mathrm{BC}")
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .* norm , label=L"\widehat{D}_\mathrm{BC}", color=tab20_color(16), linestyle="solid", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .* norm , label=L"\widehat{N}^\mathrm{KE}_{\mathrm{BT} \rightarrow \mathrm{BC}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13] .* norm , label=L"\widehat{N}^\mathrm{KE}_\mathrm{BC}", color=tab20_color(18), linestyle="dashed", linewidth=lwax)

    resid_BC_EKE = -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .- a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13]

    ax2.plot(grid.kr*Ld,resid_BC_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # BT EKE budget
    ##############################################################

    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"\widehat{T}_\mathrm{flat}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .* norm , label=L"\widehat{N}^\mathrm{KE}_{\mathrm{BC} \rightarrow \mathrm{BT}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .* norm , label=L"\widehat{N}^\mathrm{KE}_\mathrm{BT}", color=tab20_color(5), linestyle="dashed", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"\widehat{D}_\mathrm{BT}", color=tab20_color(17), linestyle="solid", linewidth=lwax)

    resid_BT_EKE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8]

    ax3.plot(grid.kr*Ld,resid_BT_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # extras
    ##############################################################


    ax1.set_xlim(0., 1.5)
    ax2.set_xlim(0., 1.5)
    ax3.set_xlim(0., 1.5)

    ax1.set_ylim(-0.25, 0.25)
    ax2.set_ylim(-0.25, 0.25)
    ax3.set_ylim(-0.25, 0.25)

    if legend_bool==true
        ax1.legend(loc="upper right", fontsize=lsize, bbox_to_anchor=(0.925, -1.35), ncols=2)
        ax2.legend(loc="upper right", fontsize=lsize, ncols=2, bbox_to_anchor=(1.1, -1.35))
        ax3.legend(loc="upper right", fontsize=lsize, ncols=2, bbox_to_anchor=(1.2, -1.35))
    end

    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    ax3.grid(alpha=0.3)



end

###################################################################################
###################################################################################

# fig = PyPlot.figure(figsize=(12,4.5))

# gs = fig.add_gridspec(2,4)
# ax1 = fig.add_subplot(gs[1,1])
# ax2 = fig.add_subplot(gs[2,1])
# ax3 = fig.add_subplot(gs[1,2])
# ax4 = fig.add_subplot(gs[2,2])
# ax5 = fig.add_subplot(gs[1,3])
# ax6 = fig.add_subplot(gs[2,3])

# ax7 = fig.add_subplot(gs[1:2, 4])

fig, axd = plt.subplot_mosaic([["ax1", "ax3", "ax5", "axnull", "ax7"],["ax2", "ax4", "ax6", "axnull", "ax7"]],
    figsize=(14,5), gridspec_kw=Dict("wspace"=>0.05, "hspace"=>0.05), width_ratios=[1,1,1,0.25,1])

axd["axnull"].axis("off")

ax1 = axd["ax1"]
ax2 = axd["ax2"]
ax3 = axd["ax3"]
ax4 = axd["ax4"]
ax5 = axd["ax5"]
ax6 = axd["ax6"]

ax7 = axd["ax7"]


# fig,ax = plt.subplots(2,3, figsize=(12,4.5))

fsize = 18
lsize = 14

lwax=2.0

###################################################################################
###################################################################################

plot_budget(ax1, ax3, ax5, 2, 1, 7, true)

plot_budget(ax2, ax4, ax6, 6, 1, 7, false)


###################################################################################
###################################################################################


for axn in [ax3;ax4;ax5;ax6]
    axn.set_yticklabels([])
end
for axn in [ax1; ax3; ax5]
    axn.set_xticklabels([])
end

for axn in [ax1;ax2;ax3;ax4;ax5;ax6]
    axn.tick_params(labelsize=lsize)
end

ax1.text(0.1, 0.9, L"\mathbf{(a)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax1.transAxes);
ax2.text(0.1, 0.9, L"\mathbf{(d)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax2.transAxes);
ax3.text(0.1, 0.9, L"\mathbf{(b)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax3.transAxes);
ax4.text(0.1, 0.9, L"\mathbf{(e)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax4.transAxes);
ax5.text(0.1, 0.9, L"\mathbf{(c)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax5.transAxes);
ax6.text(0.1, 0.9, L"\mathbf{(f)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax6.transAxes);

ax1.set_title(L"\mathrm{EAPE}", fontsize = fsize)
ax3.set_title(L"\mathrm{EKE}_\mathrm{BC}", fontsize = fsize)
ax5.set_title(L"\mathrm{EKE}_\mathrm{BT}", fontsize = fsize)

ax2.set_xlabel(L"| \boldsymbol{k} | \lambda", fontsize = fsize)
ax4.set_xlabel(L"| \boldsymbol{k} | \lambda", fontsize = fsize)
ax6.set_xlabel(L"| \boldsymbol{k} | \lambda", fontsize = fsize)

ax5.text(0.75, 0.8, L"\kappa^* = 0.25", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax5.transAxes, color="red");
ax6.text(0.75, 0.8, L"\kappa^* = 4.0", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax6.transAxes, color="red");

###################################################################################
###################################################################################

# plt.subplots_adjust(wspace=0.05, hspace=0.05)

plt.setp(ax2.get_xticklabels()[end], visible=false)
plt.setp(ax4.get_xticklabels()[end], visible=false)
# plt.setp(ax6.get_xticklabels()[end], visible=false)

ax2.set_zorder(-1)
ax4.set_zorder(-1)
ax6.set_zorder(-1)

###################################################################################
###################################################################################
# coherence plot for kappa*=4.0, beta* = 0.

tsize = 16

x = prob.grid.kr .* Ld

############################################################################################
# strong-drag f-plane
############################################################################################

mu_loc = mus[6]
beta = betas[7]

global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, 0., (0., 0.), (0., 0.))
global model_params = redef_mu_kappa_beta(model_params, mu_loc, 0., beta)

a = load(data_dir*jld_name(model_params,71.95))

#####################

ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{T}^\mathrm{D})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{N}^\mathrm{KE}_{\mathrm{BC} \rightarrow \mathrm{BT}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{N}_\mathrm{BC}^\mathrm{KE}, \, \widehat{N}^\mathrm{KE}_{\mathrm{BC} \rightarrow \mathrm{BT}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{N}_\mathrm{BC}^\mathrm{KE}, \,  \widehat{T}^\mathrm{D})", linewidth=2.0)

############################################################################################

ax7.set_xlim(x[2], 1.5)

ax7.set_ylim(0., 1.)

ax7.legend(loc="upper right", fontsize = lsize, ncols=1, bbox_to_anchor=(0.9, -0.15))

ax7.grid()

ax7.set_xlabel(L"|\boldsymbol{k}| \, \lambda", fontsize = fsize)


ax7.set_ylabel(L"\mathrm{Coherence}", fontsize = fsize)

ax7.tick_params(axis="both", labelsize=lsize)

############################################################################################

ax7.set_title(L"\mathbf{(g)} \ \kappa^* = 4.0, \, \beta^* = 0.0", fontsize=tsize)

############################################################################################


###################################################################################
###################################################################################


savefig("./JFM_BC_nrg_figs/kappas_beta0_1Dspectra.pdf",bbox_inches="tight")



function plot_budget(ax1, ax2, ax3, i, j, k, legend_bool)

    h0 = (h0s[7], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[i]

    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

    beta = betas[k]

    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

    a = load(data_dir*jld_name(model_params,71.95))


    norm = a["jld_data"]["two_layer_vBT_scale"]^-3 * Ld^-1

    ##############################################################
    # EAPE budget
    ##############################################################
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .* norm, label=L"\widehat{C}_\mathrm{BC}", color=tab20_color(1), linestyle="solid", linewidth=lwax)
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .* norm , label=L"\widehat{N}^\mathrm{EAPE}_\mathrm{BC}", color=tab20_color(19), linestyle="dashed", linewidth=lwax)
    # ax1.plot(grid.kr*Ld, 2 * a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"-2 \, D_\mathrm{BC}")
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"\widehat{T}^\mathrm{D}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)

    resid_EAPE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4]

    ax1.plot(grid.kr*Ld,resid_EAPE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # BC EKE budget
    ##############################################################
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"-\widehat{T}^\mathrm{D}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"-\widehat{T}_\mathrm{flat}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    # ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"D_\mathrm{BC}")
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .* norm , label=L"\widehat{D}_\mathrm{BC}", color=tab20_color(16), linestyle="solid", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .* norm , label=L"\widehat{N}^\mathrm{EKE}_{\mathrm{BT} \rightarrow \mathrm{BC}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13] .* norm , label=L"\widehat{N}^\mathrm{EKE}_\mathrm{BC}", color=tab20_color(14), linestyle="dashed", linewidth=lwax)

    resid_BC_EKE = -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .- a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13]

    ax2.plot(grid.kr*Ld,resid_BC_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)


    ##############################################################
    # BT EKE budget
    ##############################################################

    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"\widehat{T}_\mathrm{flat}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .* norm , label=L"\widehat{N}^\mathrm{EKE}_{\mathrm{BC} \rightarrow \mathrm{BT}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .* norm , label=L"\widehat{N}^\mathrm{EKE}_\mathrm{BT}", color=tab20_color(5), linestyle="dashed", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"\widehat{D}_\mathrm{BT}", color=tab20_color(17), linestyle="solid", linewidth=lwax)

    resid_BT_EKE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8]

    ax3.plot(grid.kr*Ld,resid_BT_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # extras
    ##############################################################


    ax1.set_xlim(0., 1.5)
    ax2.set_xlim(0., 1.5)
    ax3.set_xlim(0., 1.5)

    ax1.set_ylim(-0.25, 0.25)
    ax2.set_ylim(-0.25, 0.25)
    ax3.set_ylim(-0.25, 0.25)

    # if legend_bool==true
    #     ax1.legend(loc="upper right", fontsize=lsize, bbox_to_anchor=(0.95, -1.35), ncols=2)
    #     ax2.legend(loc="upper right", fontsize=lsize, ncols=2, bbox_to_anchor=(1.0, -1.35))
    #     ax3.legend(loc="upper right", fontsize=lsize, ncols=2, bbox_to_anchor=(1.0, -1.35))
    # end

    ax1.grid()
    ax2.grid()
    ax3.grid()

end

###################################################################################
###################################################################################

fig, axd = plt.subplot_mosaic([["ax1", "ax3", "ax5", "axnull", "ax7"],["ax2", "ax4", "ax6", "axnull", "ax7"]],
    figsize=(14,5), gridspec_kw=Dict("wspace"=>0.05, "hspace"=>0.05), width_ratios=[1,1,1,0.25,1])

axd["axnull"].axis("off")

ax1 = axd["ax1"]
ax2 = axd["ax2"]
ax3 = axd["ax3"]
ax4 = axd["ax4"]
ax5 = axd["ax5"]
ax6 = axd["ax6"]

ax7 = axd["ax7"]


# fig,ax = plt.subplots(2,3, figsize=(12,4.5))

fsize = 18
lsize = 12

lwax=2.0

###################################################################################

plot_budget(ax1, ax3, ax5, 2, 1, 8, true)

plot_budget(ax2, ax4, ax6, 2, 1, 10, false)

###################################################################################
###################################################################################

for axn in [ax3;ax4;ax5;ax6]
    axn.set_yticklabels([])
end
for axn in [ax1; ax3; ax5]
    axn.set_xticklabels([])
end

for axn in [ax1;ax2;ax3;ax4;ax5;ax6]
    axn.tick_params(labelsize=lsize)
end

ax1.text(0.1, 0.9, L"\mathbf{(a)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax1.transAxes);
ax2.text(0.1, 0.9, L"\mathbf{(d)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax2.transAxes);
ax3.text(0.1, 0.9, L"\mathbf{(b)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax3.transAxes);
ax4.text(0.1, 0.9, L"\mathbf{(e)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax4.transAxes);
ax5.text(0.1, 0.9, L"\mathbf{(c)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax5.transAxes);
ax6.text(0.1, 0.9, L"\mathbf{(f)}", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax6.transAxes);

ax1.set_title(L"\mathrm{EAPE}", fontsize = fsize)
ax3.set_title(L"\mathrm{EKE}_\mathrm{BC}", fontsize = fsize)
ax5.set_title(L"\mathrm{EKE}_\mathrm{BT}", fontsize = fsize)

ax2.set_xlabel(L"| \boldsymbol{k} | \lambda", fontsize = fsize)
ax4.set_xlabel(L"| \boldsymbol{k} | \lambda", fontsize = fsize)
ax6.set_xlabel(L"| \boldsymbol{k} | \lambda", fontsize = fsize)


###################################################################################
###################################################################################

ax5.text(0.75, 0.8, L"\beta^* = 0.25", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax5.transAxes, color="red");
ax6.text(0.75, 0.8, L"\beta^* = 0.75", fontsize=fsize, horizontalalignment="center",verticalalignment="center", transform = ax6.transAxes, color="red");

plt.setp(ax2.get_xticklabels()[end], visible=false)
plt.setp(ax4.get_xticklabels()[end], visible=false)
# plt.setp(ax6.get_xticklabels()[end], visible=false)

ax2.set_zorder(-1)
ax4.set_zorder(-1)
ax6.set_zorder(-1)

###################################################################################
###################################################################################
# coherence plot for kappa*=4.0, beta* = 0.

tsize = 16
# fsize = 16
# lsize = 11

x = prob.grid.kr .* Ld

############################################################################################
# strong-drag f-plane
############################################################################################

mu_loc = mus[2]
beta = betas[10]

global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, 0., (0., 0.), (0., 0.))
global model_params = redef_mu_kappa_beta(model_params, mu_loc, 0., beta)

a = load(data_dir*jld_name(model_params,71.95))

#####################

ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{T}^\mathrm{D})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{N}^\mathrm{EKE}_{\mathrm{BC} \rightarrow \mathrm{BT}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{N}_\mathrm{BC}^\mathrm{EKE}, \, \widehat{N}^\mathrm{EKE}_{\mathrm{BC} \rightarrow \mathrm{BT}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{N}_\mathrm{BC}^\mathrm{EKE}, \,  \widehat{T}^\mathrm{D})", linewidth=2.0)

############################################################################################

ax7.set_xlim(x[2], 1.5)

ax7.set_ylim(0., 1.)

# ax7.legend(loc="upper right", fontsize = lsize, ncols=2, bbox_to_anchor=(1.3, -0.15))

ax7.grid()

ax7.set_xlabel(L"|\boldsymbol{k}| \, \lambda", fontsize = fsize)


ax7.set_ylabel(L"\mathrm{Coherence}", fontsize = fsize)

ax7.tick_params(axis="both", labelsize=lsize)

############################################################################################

ax7.set_title(L"\mathbf{(g)} \ \kappa^* = 0.25, \, \beta^* = 0.75", fontsize=tsize)

############################################################################################


###################################################################################
###################################################################################


savefig("./JFM_BC_nrg_figs/kappa0p25_betas_1Dspectra.pdf",bbox_inches="tight")


i=7
j=1
k=2


h0 = (h0s[i], 0.0)
kappa_loc = kappas[j]
mu_loc = mus[k]

global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
global topo_PV, eta = define_topo(model_params)
global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

beta = betas[7]
global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

a = load(data_dir*jld_name(model_params,71.95))


###################################################################################
###################################################################################
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html
fig, axs = plt.subplots(2, 4, layout="constrained", figsize=(13, 7)) # , width_ratios = [1.2,1.,1.,1.])
gridspec = axs[1, 1].get_subplotspec().get_gridspec()



# clear the left column for the subfigure:
# for a in axs[:, 1]
#     a.remove()
# end

###########################################################################
############################################################################
# fig, ax = plt.subplots(1,3, figsize=(10,3))
# fig.tight_layout(pad=-2.0)

ax1=axs[1,2]; ax2=axs[1,3]; ax3=axs[1,4];

tsize = 20
fsize = 18
lsize = 12

x = prob.grid.kr .* Ld
y = prob.grid.l .* Ld

x = append!(x[:,1], [x[end] + x[end] - x[end-1]])
# y = append!(y[1,:], [y[end] + y[end] - y[end-1]])

y = fftshift(y)
y = append!(y[1,:], [20.48])

lim_range = 30

lim = 1.0 * maximum([abs.(grid.Krsq .* a["jld_data"]["CBC"]); abs.(grid.Krsq .* a["jld_data"]["T_D"]); abs.(grid.Krsq .* a["jld_data"]["NL_BC_EAPE"])])

lim1 = lim2 = lim3 = lim
########################################################################################################

# lim1 = maximum(abs.(a["jld_data"]["CBC"]))

pc1 = ax1.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["CBC"],2)', vmin=-lim1, vmax=lim1, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc1.set_edgecolor("face")
# plt.colorbar(pc1, location="bottom")

ax1.set_xlim(x[1],x[lim_range])
ax1.set_ylim(0,y[512+lim_range])

# PyPlot.subplots_adjust(wspace=0, hspace=0)


########################################################################################################

# lim2 = maximum(abs.(a["jld_data"]["T_D"]))

pc2 = ax2.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["T_D"],2)', vmin=-lim2, vmax=lim2, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc2.set_edgecolor("face")
# plt.colorbar(pc1, location="bottom")

ax2.set_xlim(x[1],x[lim_range])
ax2.set_ylim(0,y[512+lim_range])

########################################################################################################

# lim3 = maximum(abs.(a["jld_data"]["NL_BC_EAPE"]))

pc3 = ax3.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["NL_BC_EAPE"],2)', vmin=-lim3, vmax=lim3, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc3.set_edgecolor("face")
# plt.colorbar(pc1, location="bottom")

ax3.set_xlim(x[1],x[lim_range])
ax3.set_ylim(0,y[512+lim_range])

########################################################################################################

ax2.set_yticklabels([])
ax3.set_yticklabels([])

ax2.set_title(L"\mathbf{(b)} \ \kappa^* = 0.25, \, \beta^* = 0.0", fontsize=tsize)

ax1.text(0.025, 0.975, L"\widehat{C}_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax1.transAxes);
ax2.text(0.025, 0.95, L"\widehat{T}^\mathrm{D}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax2.transAxes);
ax3.text(0.025, 0.975, L"\widehat{N}^\mathrm{PE}_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax3.transAxes);

# ax1.grid()
# ax2.grid()
# ax3.grid()


############################################################################
###########################################################################
# make the subfigure in the empty gridspec slots:

subfig = fig.add_subfigure(gridspec[1, 1])

axul = axs[1,1] # subfig.subplots(1, 1)

plot_zetaBT(a, prob, axul)

axul.set_title(L"\mathbf{(a)} \ \mathrm{EKE}_\mathrm{BT}", fontsize=tsize)


############################################################################
###########################################################################

beta = betas[10]
global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

a = load(data_dir*jld_name(model_params,71.95))

ax1=axs[2,2]; ax2=axs[2,3]; ax3=axs[2,4];

lim = 0.25 * maximum([abs.(grid.Krsq .* a["jld_data"]["CBC"]); abs.(grid.Krsq .* a["jld_data"]["T_D"]); abs.(grid.Krsq .* a["jld_data"]["NL_BC_EAPE"])])

lim1 = lim2 = lim3 = lim
########################################################################################################

# lim1 = maximum(abs.(a["jld_data"]["CBC"]))

pc1 = ax1.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["CBC"],2)', vmin=-lim1, vmax=lim1, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc1.set_edgecolor("face")

ax1.set_xlim(x[1],x[lim_range])
ax1.set_ylim(0,y[512+lim_range])

########################################################################################################

# lim2 = maximum(abs.(a["jld_data"]["T_D"]))

pc2 = ax2.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["T_D"],2)', vmin=-lim2, vmax=lim2, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc2.set_edgecolor("face")
# plt.colorbar(pc1, location="bottom")

ax2.set_xlim(x[1],x[lim_range])
ax2.set_ylim(0,y[512+lim_range])

########################################################################################################

# lim3 = maximum(abs.(a["jld_data"]["NL_BC_EAPE"]))

pc3 = ax3.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["NL_BC_EAPE"],2)', vmin=-lim3, vmax=lim3, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc3.set_edgecolor("face")
# plt.colorbar(pc1, location="bottom")

ax3.set_xlim(x[1],x[lim_range])
ax3.set_ylim(0,y[512+lim_range])

########################################################################################################

ax2.set_yticklabels([])
ax3.set_yticklabels([])

ax2.set_title(L"\mathbf{(d)} \ \kappa^* = 0.25, \, \beta^* = 0.75", fontsize=tsize)

ax1.text(0.025, 0.975, L"\widehat{C}_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax1.transAxes);
ax2.text(0.025, 0.95, L"\widehat{T}^\mathrm{D}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax2.transAxes);
ax3.text(0.025, 0.975, L"\widehat{N}^\mathrm{PE}_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax3.transAxes);

# ax1.grid()
# ax2.grid()
# ax3.grid()



############################################################################
###########################################################################

subfig = fig.add_subfigure(gridspec[2, 1])

axll = axs[2,1] #subfig.subplots(1, 1)

plot_zetaBT(a, prob, axll)

axll.set_title(L"\mathbf{(c)} \ \mathrm{EKE}_\mathrm{BT}", fontsize=tsize)


############################################################################
###########################################################################
# make the subfigure in the empty gridspec slots:



# for axn in ax[4:end]
#     axn.set_yticklabels([])
# end
# for axn in [ax[1:2]; ax[4:5]; ax[7:8]]
#     axn.set_xticklabels([])
# end

axs[1,2].tick_params(labelsize=lsize)
axs[1,3].tick_params(labelsize=lsize)
axs[1,4].tick_params(labelsize=lsize)
axs[2,2].tick_params(labelsize=lsize)
axs[2,3].tick_params(labelsize=lsize)
axs[2,4].tick_params(labelsize=lsize)

axll.tick_params(labelsize=lsize)
axul.tick_params(labelsize=lsize)

axul.set_ylabel(L"y / \lambda \, 2 \pi ", fontsize=fsize)
axll.set_ylabel(L"y / \lambda \, 2 \pi", fontsize=fsize)

axul.set_xlabel(L"x / \lambda \, 2 \pi ", fontsize=fsize)
axll.set_xlabel(L"x / \lambda \, 2 \pi ", fontsize=fsize)

axs[1,2].set_xlabel(L"k_{x} \lambda ", fontsize=fsize)
axs[1,3].set_xlabel(L"k_{x} \lambda ", fontsize=fsize)
axs[1,4].set_xlabel(L"k_{x} \lambda ", fontsize=fsize)
axs[2,2].set_xlabel(L"k_{x} \lambda ", fontsize=fsize)
axs[2,3].set_xlabel(L"k_{x} \lambda ", fontsize=fsize)
axs[2,4].set_xlabel(L"k_{x} \lambda ", fontsize=fsize)


axs[1,2].set_ylabel(L"k_{y} \lambda ", fontsize=fsize)
axs[2,2].set_ylabel(L"k_{y} \lambda ", fontsize=fsize)

# subfig.set_facecolor('0.75')
# for ax in axsLeft:
#     pc = example_plot(ax)
# subfig.suptitle('Left plots', fontsize='x-large')
# subfig.colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

# fig.suptitle('Figure suptitle', fontsize='xx-large')
# plt.show()


savefig("./JFM_BC_nrg_figs/zetaBT_2Dspectra.pdf",bbox_inches="tight")























