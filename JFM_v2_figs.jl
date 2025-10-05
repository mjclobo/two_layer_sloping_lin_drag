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
####################################################################
##
####################################################################
function remove_isotropic_mean(arr_in, grid)
    # arr_in: an nkr X nl array that is output of rfft
    # note that we only want real part of this

    dev = grid.device
    T = eltype(grid)
    A = device_array(dev)
    
    dk = 2*pi/grid.Lx; dl = 2*pi/grid.Ly;
    
    dkr = sqrt(dk^2 + dl^2)
    
    wv = @. sqrt(grid.kr^2 + grid.l^2)
    
    iso = zeros(dev, T, (length(grid.kr)))
    
    for i in range(1,length(grid.kr))
        # find 2D index values for a wavenumber magnitude
        if i==length(grid.kr)
            fkr = CUDA.@allowscalar  @. (wv>=grid.kr[i]) & (wv<=grid.kr[i]+dkr)
        else
            fkr = CUDA.@allowscalar  @. (wv>=grid.kr[i]) & (wv<grid.kr[i+1])
        end
        
        if sum(fkr) > 0
            CUDA.@allowscalar iso[i] = sum(real(arr_in[fkr])) # this is average over all combinations of k_x and k_y that are the same, isotropic k
        end
        
    end

    return iso
end

i = 2
j=1
k=7

h0 = (h0s[7], 0.0)
kappa_loc = kappas[j]
mu_loc = mus[i]

global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
global topo_PV, eta = define_topo(model_params)
global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

beta = betas[k]

global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

a = load(data_dir*jld_name_2L(model_params,71.95))



iso_norm = remove_isotropic_mean(ones(size(a["jld_data"]["CBC"])), grid);



####################################################################
##
####################################################################


function plot_budget(ax1, ax2, ax3, i, j, k, legend_bool, ylims)

    h0 = (h0s[k], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[i]

    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

    beta = betas[7]

    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

    a = load(data_dir*jld_name_2L(model_params,71.95))


    norm = a["jld_data"]["two_layer_vBT_scale"]^-3 * Ld^-1 .* iso_norm

    ##############################################################
    # EAPE budget
    ##############################################################
    # trans_resid = -(a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .+ 2 .* a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13] )
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .* norm, label=L"\widehat{P}", color=tab20_color(1), linestyle="solid", linewidth=lwax)
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .* norm , label=L"\widehat{R}^\mathrm{PE}_\mathrm{BC}", color=tab20_color(19), linestyle="dashed", linewidth=lwax)
    # ax1.plot(grid.kr*Ld, 2 * a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"-2 \, D_\mathrm{BC}")
    ax1.plot(grid.kr*Ld,a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"\widehat{T}^\mathrm{W}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)

    resid_EAPE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4]

    ax1.plot(grid.kr*Ld,resid_EAPE .* norm, color="black", alpha=1.0, linewidth=0.5)

    # if i==2
    #     # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #     # axins = ax1.inset_axes(ax[1], width="30%", height="40%")
    #     axins = ax1.inset_axes([0.125, 0.6, 1-0.125, 0.2])

    #     axins.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .* norm, label=L"\widehat{C}_\mathrm{BC}", color=tab20_color(1), linestyle="solid", linewidth=lwax)
    #     axins.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .* norm , label=L"\widehat{N}^\mathrm{PE}_\mathrm{BC}", color=tab20_color(19), linestyle="dashed", linewidth=lwax)
    #     # axins.plot(grid.kr*Ld, 2 * a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"-2 \, D_\mathrm{BC}")
    #     axins.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"\widehat{T}^\mathrm{D}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)

    #     axins.plot(grid.kr*Ld,resid_EAPE .* norm, color="black", alpha=1.0, linewidth=0.5)

    #     axins.set_xlim(0.125, 1.5)
    #     axins.set_ylim(-0.005, 0.005)

    #     axins.set_yticklabels([])
    #     axins.set_xticklabels([])

    # end


    ##############################################################
    # BC EKE budget
    ##############################################################
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"-\widehat{T}^\mathrm{W}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"\widehat{T}^\mathrm{L}_{\mathrm{BT} \rightarrow \mathrm{BC}}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    # ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"D_\mathrm{BC}")
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .* norm , label=L"\widehat{D}_\mathrm{BC}", color=tab20_color(16), linestyle="solid", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .* norm , label=L"\widehat{T}^\mathrm{N}_{\mathrm{BT} \rightarrow \mathrm{BC}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13] .* norm , label=L"\widehat{R}^\mathrm{KE}_\mathrm{BC}", color=tab20_color(18), linestyle="dashed", linewidth=lwax)

    resid_BC_EKE = -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .- a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13]

    ax2.plot(grid.kr*Ld,resid_BC_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # BT EKE budget
    ##############################################################

    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"\widehat{T}^\mathrm{L}_{\mathrm{BC} \rightarrow \mathrm{BT}}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .* norm , label=L"\widehat{T}^\mathrm{N}_{\mathrm{BC} \rightarrow \mathrm{BT}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .* norm , label=L"\widehat{R}^\mathrm{KE}_\mathrm{BT}", color=tab20_color(5), linestyle="dashed", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"\widehat{D}_\mathrm{BT}", color=tab20_color(17), linestyle="solid", linewidth=lwax)

    resid_BT_EKE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8]

    ax3.plot(grid.kr*Ld,resid_BT_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # extras
    ##############################################################


    ax1.set_xlim(0., 1.5)
    ax2.set_xlim(0., 1.5)
    ax3.set_xlim(0., 1.5)

    # ax1.set_ylim(-0.25, 0.25)
    # ax2.set_ylim(-0.25, 0.25)
    # ax3.set_ylim(-0.25, 0.25)

    ax1.set_ylim(ylims[1], ylims[2])
    ax2.set_ylim(ylims[1], ylims[2])
    ax3.set_ylim(ylims[1], ylims[2])

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

plot_budget(ax1, ax3, ax5, 2, 1, 7, true, [-1.25, 1.25])

plot_budget(ax2, ax4, ax6, 6, 1, 7, false, [-10, 10])


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

a = load(data_dir*jld_name_2L(model_params,71.95))

#####################

ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{T}^\mathrm{W})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{T}^\mathrm{N}_{\mathrm{BT} \rightarrow \mathrm{BC}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{R}_\mathrm{BC}^\mathrm{KE}, \, \widehat{T}^\mathrm{N}_{\mathrm{BT} \rightarrow \mathrm{BC}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{R}_\mathrm{BC}^\mathrm{KE}, \,  \widehat{T}^\mathrm{W})", linewidth=2.0)

############################################################################################

ax7.set_xlim(x[2], 1.5)

ax7.set_ylim(0., 1.)

ax7.legend(loc="upper right", fontsize = lsize, ncols=1, bbox_to_anchor=(0.9675, -0.15))

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

function plot_budget(ax1, ax2, ax3, i, j, k, legend_bool, ylims)

    h0 = (h0s[7], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[i]

    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

    beta = betas[k]

    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

    a = load(data_dir*jld_name_2L(model_params,71.95))


    norm = a["jld_data"]["two_layer_vBT_scale"]^-3 * Ld^-1 .* iso_norm

    ##############################################################
    # EAPE budget
    ##############################################################
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .* norm, label=L"\widehat{P}", color=tab20_color(1), linestyle="solid", linewidth=lwax)
    ax1.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .* norm , label=L"\widehat{R}^\mathrm{PE}_\mathrm{BC}", color=tab20_color(19), linestyle="dashed", linewidth=lwax)

    ax1.plot(grid.kr*Ld,a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"\widehat{T}^\mathrm{W}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)

    resid_EAPE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,5] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,14] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4]

    ax1.plot(grid.kr*Ld,resid_EAPE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # BC EKE budget
    ##############################################################
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .* norm, label=L"-\widehat{T}^\mathrm{W}", color=tab20_color(8), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"\widehat{T}^\mathrm{L}_{\mathrm{BT} \rightarrow \mathrm{BC}}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    # ax2.plot(grid.kr*Ld, -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"D_\mathrm{BC}")
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .* norm , label=L"\widehat{D}_\mathrm{BC}", color=tab20_color(16), linestyle="solid", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .* norm , label=L"\widehat{T}^\mathrm{N}_{\mathrm{BT} \rightarrow \mathrm{BC}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax2.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13] .* norm , label=L"\widehat{R}^\mathrm{KE}_\mathrm{BC}", color=tab20_color(18), linestyle="dashed", linewidth=lwax)

    resid_BC_EKE = -a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,4] .- a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,9] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,12] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,13]

    ax2.plot(grid.kr*Ld,resid_BC_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # BT EKE budget
    ##############################################################

    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .* norm, label=L"\widehat{T}^\mathrm{L}_{\mathrm{BC} \rightarrow \mathrm{BT}}", color=tab20_color(2), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .* norm , label=L"\widehat{T}^\mathrm{N}_{\mathrm{BC} \rightarrow \mathrm{BT}}", color=tab20_color(12), linestyle="dashdot", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .* norm , label=L"\widehat{R}^\mathrm{KE}_\mathrm{BT}", color=tab20_color(5), linestyle="dashed", linewidth=lwax)
    ax3.plot(grid.kr*Ld, a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8] .* norm , label=L"\widehat{D}_\mathrm{BT}", color=tab20_color(17), linestyle="solid", linewidth=lwax)

    resid_BT_EKE = a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,6] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,11] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,10] .+ a["jld_data"]["two_layer_kspace_modal_nrg_budget"][:,8]

    ax3.plot(grid.kr*Ld,resid_BT_EKE .* norm, color="black", alpha=1.0, linewidth=0.5)

    ##############################################################
    # extras
    ##############################################################


    ax1.set_xlim(0., 1.5)
    ax2.set_xlim(0., 1.5)
    ax3.set_xlim(0., 1.5)

    ax1.set_ylim(ylims[1], ylims[2])
    ax2.set_ylim(ylims[1], ylims[2])
    ax3.set_ylim(ylims[1], ylims[2])

    if legend_bool==true
        ax1.legend(loc="upper right", fontsize=lsize, bbox_to_anchor=(0.925, -1.35), ncols=2)
        ax2.legend(loc="upper right", fontsize=lsize, ncols=2, bbox_to_anchor=(1.1, -1.35))
        ax3.legend(loc="upper right", fontsize=lsize, ncols=2, bbox_to_anchor=(1.2, -1.35))
    end

    ax1.grid()
    ax2.grid()
    ax3.grid()

end

###################################################################################
###################################################################################

# fig, axd = plt.subplot_mosaic([["ax1", "ax3", "ax5", "axnull", "ax7"],["ax2", "ax4", "ax6", "axnull", "ax7"]],
#     figsize=(14,5), gridspec_kw=Dict("wspace"=>0.05, "hspace"=>0.05), width_ratios=[1,1,1,0.25,1])

# axd["axnull"].axis("off")

# ax1 = axd["ax1"]
# ax2 = axd["ax2"]
# ax3 = axd["ax3"]
# ax4 = axd["ax4"]
# ax5 = axd["ax5"]
# ax6 = axd["ax6"]

# ax7 = axd["ax7"]


# # fig,ax = plt.subplots(2,3, figsize=(12,4.5))

# fsize = 18
# lsize = 12

# lwax=2.0

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

plot_budget(ax1, ax3, ax5, 2, 1, 8, true, [-2.5, 2.5])

plot_budget(ax2, ax4, ax6, 2, 1, 10, false, [-10, 10])

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

a = load(data_dir*jld_name_2L(model_params,71.95))

#####################

ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{T}^\mathrm{W})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_DBC_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{D}_\mathrm{BC}, \, \widehat{T}^\mathrm{N}_{\mathrm{BT} \rightarrow \mathrm{BC}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_NLBC2BT"] , prob.grid), label=L"\mathrm{C} ( \widehat{R}_\mathrm{BC}^\mathrm{KE}, \, \widehat{T}^\mathrm{N}_{\mathrm{BT} \rightarrow \mathrm{BC}})", linewidth=2.0)
ax7.plot(x, isotropic_mean(a["jld_data"]["coh_NLBCEKE_TD"] , prob.grid), label=L"\mathrm{C} ( \widehat{R}_\mathrm{BC}^\mathrm{KE}, \,  \widehat{T}^\mathrm{W})", linewidth=2.0)

############################################################################################

ax7.set_xlim(x[2], 1.5)

ax7.set_ylim(0., 1.)

ax7.legend(loc="upper right", fontsize = lsize, ncols=1, bbox_to_anchor=(0.9675, -0.15))

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

a = load(data_dir*jld_name_2L(model_params,71.95))


###################################################################################
###################################################################################
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html
fig, ax = plt.subplots(3, 3, layout="constrained", figsize=(9.5, 7), height_ratios=[1., 0.005, 1.]) # , width_ratios = [1.2,1.,1.,1.])

ax1=ax[1]; ax2=ax[4]; ax3=ax[7];
ax[2].axis("off");
ax[5].axis("off");
ax[8].axis("off");
ax4=ax[3]; ax5=ax[6]; ax6=ax[9];

# clear the left column for the subfigure:
# for a in axs[:, 1]
#     a.remove()
# end

###########################################################################
############################################################################
# fig, ax = plt.subplots(1,3, figsize=(10,3))
# fig.tight_layout(pad=-2.0)

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


ax1.text(0.025, 0.975, L"\mathbf{(a)} \ \widehat{P}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax1.transAxes);
ax2.text(0.025, 0.95, L"\mathbf{(b)} \ \widehat{T}^\mathrm{W}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax2.transAxes);
ax3.text(0.025, 0.975, L"\mathbf{(c)} \ \widehat{R}^\mathrm{PE}_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax3.transAxes);

# ax1.grid()
# ax2.grid()
# ax3.grid()

ax2.text(0.5, 1.115, L"\kappa^* = 0.25, \, \beta^* = 0.0", fontsize=tsize, horizontalalignment="center",verticalalignment="center", transform = ax2.transAxes,
    bbox=Dict("facecolor"=>"none", "edgecolor"=>"black"));



############################################################################
###########################################################################

beta = betas[10]
global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

a = load(data_dir*jld_name_2L(model_params,71.95))


lim = 0.25 * maximum([abs.(grid.Krsq .* a["jld_data"]["CBC"]); abs.(grid.Krsq .* a["jld_data"]["T_D"]); abs.(grid.Krsq .* a["jld_data"]["NL_BC_EAPE"])])

lim1 = lim2 = lim3 = lim
########################################################################################################

# lim1 = maximum(abs.(a["jld_data"]["CBC"]))

pc4 = ax4.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["CBC"],2)', vmin=-lim1, vmax=lim1, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc4.set_edgecolor("face")

ax4.set_xlim(x[1],x[lim_range])
ax4.set_ylim(0,y[512+lim_range])

########################################################################################################

# lim2 = maximum(abs.(a["jld_data"]["T_D"]))

pc5 = ax5.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["T_D"], 2)', vmin=-lim2, vmax=lim2, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc5.set_edgecolor("face")
# plt.colorbar(pc1, location="bottom")

ax5.set_xlim(x[1],x[lim_range])
ax5.set_ylim(0,y[512+lim_range])

########################################################################################################

# lim3 = maximum(abs.(a["jld_data"]["NL_BC_EAPE"]))

pc6 = ax6.pcolormesh(x, y, fftshift(grid.Krsq .* a["jld_data"]["NL_BC_EAPE"],2)', vmin=-lim3, vmax=lim3, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc6.set_edgecolor("face")
# plt.colorbar(pc1, location="bottom")

ax6.set_xlim(x[1],x[lim_range])
ax6.set_ylim(0,y[512+lim_range])

########################################################################################################

ax5.set_yticklabels([])
ax6.set_yticklabels([])

ax5.text(0.5, 1.115, L" \kappa^* = 0.25, \, \beta^* = 0.75", fontsize=tsize, horizontalalignment="center",verticalalignment="center", transform = ax5.transAxes,
    bbox=Dict("facecolor"=>"none", "edgecolor"=>"black"));

ax4.text(0.025, 0.975, L"\mathbf{(d)} \ \widehat{P}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax4.transAxes);
ax5.text(0.025, 0.95, L"\mathbf{(e)} \ \widehat{T}^\mathrm{W}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax5.transAxes);
ax6.text(0.025, 0.975, L"\mathbf{(f)} \ \widehat{R}^\mathrm{PE}_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax6.transAxes);

# ax1.grid()
# ax2.grid()
# ax3.grid()





############################################################################
###########################################################################
# make the subfigure in the empty gridspec slots:



# for axn in ax[4:end]
#     axn.set_yticklabels([])
# end
# for axn in [ax[1:2]; ax[4:5]; ax[7:8]]
#     axn.set_xticklabels([])
# end

for axn in ax
    axn.tick_params(labelsize=lsize)
    axn.set_xlabel(L"k_{x} \lambda ", fontsize=fsize)
end

ax1.set_ylabel(L"k_{y} \lambda ", fontsize=fsize)
ax4.set_ylabel(L"k_{y} \lambda ", fontsize=fsize)

# subfig.set_facecolor('0.75')
# for ax in axsLeft:
#     pc = example_plot(ax)
# subfig.suptitle('Left plots', fontsize='x-large')
# subfig.colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

# fig.suptitle('Figure suptitle', fontsize='xx-large')
# plt.show()


savefig("./JFM_BC_nrg_figs/2Dspectra.pdf",bbox_inches="tight")












h0 = (0., 0.0)
kappa_loc = kappas[1]
mu_loc = mus[2]

global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
global topo_PV, eta = define_topo(model_params)
global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

beta = β = betas[10]

global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)

# data_dir_omega = "/scratch/cimes/ml1994/QG/julia/data/JFM_omega_diags_data/"
data_dir_omega = "/scratch/cimes/ml1994/QG/julia/data/JFM_lp_omega_diags_data/"
data_dir_omega = "/scratch/cimes/ml1994/QG/julia/data/JFM_2D_omega_diags_data/"

a = load(data_dir_omega*jld_name(model_params,71.95))

w_names = [L"full", L"\nabla^2 \, J(\psi_{2}, \psi_{1})", L"J(\psi_{2}, f)", L"J(\psi_{2}, \zeta_{2})", L"J(\psi_{1}, f)",
            L"J(\psi_{1}, \zeta_{1})", L"J(U_{1}y, \zeta_{1})", L"w_{b}", L"\nabla^2 \, J(\psi_{2}, S_{3/2} y)"];
TW_names= ["TW_full_h", "TW_p2p1_h", "TW_p2f_h", "TW_p2z2_h", "TW_p1f_h", "TW_p1z1_h", "TW_U1z1_h", "TW_wb_h", "TW_S32_h"]

function turn_off_ax(ax; ax1_bool=false)
    if ax1_bool==true
        # Turn off the top, right, and left spines
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        
        # Turn off ticks and labels for the top, right, and left axes
        ax.tick_params(top=false, right=false)
        ax.tick_params(labeltop=false, labelright=false)
    
        # Ensure the bottom axis and its labels are visible
        # ax.tick_params(bottom=true, labelbottom=true)

    else
        # Turn off the top, right, and left spines
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        
        # Turn off ticks and labels for the top, right, and left axes
        ax.tick_params(top=false, right=false, left=false)
        ax.tick_params(labeltop=false, labelright=false, labelleft=false)
    
        # Ensure the bottom axis and its labels are visible
        ax.tick_params(bottom=true, labelbottom=true)
    end
end







fig, ax = plt.subplots(1, 4, figsize=(12.5,3))

fig,tight_layout(pad=-0.5)

tsize = 20
fsize = 20
lsize = 14

x = prob.grid.kr .* Ld
y = prob.grid.l .* Ld

x = append!(x[:,1], [x[end] + x[end] - x[end-1]])
# y = append!(y[1,:], [y[end] + y[end] - y[end-1]])

y = fftshift(y)
y = append!(y[:], [20.48])

lim_range = 30

lim = maximum(sqrt.(grid.Krsq) .* abs.(real.(a["jld_data"]["TW_full_h"])))/5

# ax[1].pcolormesh((sqrt.(grid.Krsq) .* real.(a["jld_data"][names[2]]))', vmin=-lim, vmax=lim, cmap=PyPlot.cm.bwr)
# ax[1].set_title(L"\nabla^2 \, J(\psi_{2}, \psi_{1})")

# ax[2].pcolormesh((sqrt.(grid.Krsq) .* real.(a["jld_data"][TW_names[3]] .+ a["jld_data"][TW_names[5]]))', vmin=-lim, vmax=lim, cmap=PyPlot.cm.bwr)
# ax[2].set_title(L"J(\tau, f)")

ax[1].pcolormesh(x, y, fftshift(sqrt.(grid.Krsq) .* real.(a["jld_data"][TW_names[2]] .+ a["jld_data"][TW_names[4]] .+ a["jld_data"][TW_names[6]]), 2)', vmin=-lim, vmax=lim, cmap=PyPlot.cm.bwr, linewidth=0., rasterized=true)

ax[2].pcolormesh(x, y, fftshift(sqrt.(grid.Krsq) .* real.(a["jld_data"][TW_names[8]]), 2)', vmin=-lim, vmax=lim, cmap=PyPlot.cm.bwr, linewidth=0., rasterized=true)

ax[3].pcolormesh(x, y, fftshift(sqrt.(grid.Krsq) .* real.(a["jld_data"][TW_names[7]] .+ a["jld_data"][TW_names[9]]), 2)', vmin=-lim, vmax=lim, cmap=PyPlot.cm.bwr, linewidth=0., rasterized=true)


# sum_tot = zeros(size(real.(a["jld_data"][names[1]])))
# for name in names[2:end]
#     sum_tot .+= real.(a["jld_data"][name])
# end

ax[4].pcolormesh(x, y, fftshift(sqrt.(grid.Krsq) .* real.(a["jld_data"][TW_names[1]]), 2)', vmin=-lim, vmax=lim, cmap=PyPlot.cm.bwr, linewidth=0., rasterized=true)
# ax[6].pcolormesh((sqrt.(grid.Krsq) .* (real.(a["jld_data"][names[1]]) .- sum_tot))', vmin=-lim, vmax=lim, cmap=PyPlot.cm.bwr)



for axn in ax
    axn.set_xlim(x[1],x[lim_range])
    axn.set_ylim(0,y[512+lim_range])
    axn.tick_params(labelsize=lsize)
    axn.set_xlabel(L"k_{x} \lambda ", fontsize=fsize)
end

for axn in ax[2:end]
    axn.set_yticklabels([])
end

ax[1].set_ylabel(L"k_{y} \lambda ", fontsize=fsize)


ax[1].text(0.025, 0.975, L"\mathbf{(a)} \ \ \widehat{T}^\mathrm{W} [ \mathrm{J}(\tau, \nabla^2 \psi) ]", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax[1].transAxes);
ax[2].text(0.025, 0.95, L"\mathbf{(b)} \ \ \widehat{T}^\mathrm{W} [ w_\mathrm{b} ]", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax[2].transAxes);
ax[3].text(0.025, 0.975, L"\mathbf{(c)} \ \ \widehat{T}^\mathrm{W} [ \mathrm{J}(U_{1} y, \nabla^2 \psi) ]", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax[3].transAxes);
ax[4].text(0.025, 0.975, L"\mathbf{(d)} \ \ \widehat{T}^\mathrm{W} \ \mathrm{total} ", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax[4].transAxes);

savefig("./JFM_BC_nrg_figs/TW_decomp.pdf",bbox_inches="tight", transparent=true)





function calc_CBC_DBC_EKE(ψ)
    rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, nlayers), [1, 2]; flags=FFTW.MEASURE)
    rfftplan = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, 1), [1, 2]; flags=FFTW.MEASURE)

    # parameters
    gr = gp(rho[1:2],rho0,g)

    Ld = sqrt(gr * sum(H)) / (2 * f0)
    # ∂yηb = topographic_pv_gradient[2] / (f0 / H[end])

    ψh = deepcopy(vars.ψh)

    mul!(ψh, rfftplanlayered, deepcopy(ψ))

    u = deepcopy(ψ)
    v = deepcopy(ψ)

    ldiv!(u, rfftplanlayered, -im * grid.l  .* ψh)
    ldiv!(v, rfftplanlayered, im * grid.kr  .* ψh)

    U₁, U₂, = view(params.U, :, :, 1), view(params.U, :, :, 2)
    u1, v1 = view(u, :, :, 1), view(v, :, :, 1)
    u2, v2 = view(u, :, :, 2), view(v, :, :, 2)

    ψ1, ψ2 = view(ψ, :, :, 1), view(ψ, :, :, 2)
    ψ1h, ψ2h = view(ψh, :, :, 1), view(ψh, :, :, 2)

    # # calculating terms used in budget
    ψ₁h, ψ₂h = view(ψh, :, :, 1), view(ψh, :, :, 2)

    # nonlinear terms
    ζh = -grid.Krsq .* ψh

    ζ = deepcopy(v)
    ldiv!(ζ, rfftplanlayered, -grid.Krsq .* ψh)

    ζ1, ζ2 = view(ζ, :, :, 1), view(ζ, :, :, 2)

    ∂xζ = deepcopy(ψ)
    ldiv!(∂xζ, rfftplanlayered, im .* grid.kr .* ζh)

    ∂yζ = deepcopy(ψ)
    ldiv!(∂yζ, rfftplanlayered, im .* grid.l .* ζh)

    ∂xζ1, ∂xζ2 = view(∂xζ, :, :, 1), view(∂xζ, :, :, 2)
    ∂yζ1, ∂yζ2 = view(∂yζ, :, :, 1), view(∂yζ, :, :, 2)

    ##
    J_ψ1_f = @. v1 * β
    J_ψ2_f = @. v2 * β

    J_ψ1_fh = deepcopy(ψh[:,:,1])
    mul2D!(J_ψ1_fh, rfftplan, J_ψ1_f)

    J_ψ2_fh = deepcopy(ψh[:,:,1])
    mul2D!(J_ψ2_fh, rfftplan, J_ψ2_f)

    ## J(ψ1, ζ1) & J(ψ2, ζ2)
    J_ψ1_ζ1 = @. u1 * ∂xζ1 + v1 * ∂yζ1
    J_ψ2_ζ2 = @. u2 * ∂xζ2 + v2 * ∂yζ2

    J_ψ1_ζ1h = deepcopy(ψh[:,:,1])
    mul2D!(J_ψ1_ζ1h, rfftplan, J_ψ1_ζ1)

    J_ψ2_ζ2h = deepcopy(ψh[:,:,1])
    mul2D!(J_ψ2_ζ2h, rfftplan, J_ψ2_ζ2)

    ##
    J_ψ2_ψ1 = @. v1 * u2 - u1 * v2

    J_ψ2_ψ1h = deepcopy(ψh[:,:,1])
    mul2D!(J_ψ2_ψ1h, rfftplan, J_ψ2_ψ1)

    ∇2J_ψ2_ψ1h = - grid.Krsq .* J_ψ2_ψ1h

    ##
    v2h = im .* grid.kr .* ψ2h
    J_ψ2_S32h = v2h .* f0 .* U₁ ./ gr

    ∇2J_ψ2_S32h = - grid.Krsq .* J_ψ2_S32h

    ##
    U1_∂xζ1h = deepcopy(ψh[:,:,1])
    mul2D!(U1_∂xζ1h, rfftplan, U₁ .* ∂xζ1 )

    ##
    w_b = @. mus[2] * H[2] * ζ2 / f0

    w_bh = deepcopy(ψh[:,:,1])
    mul2D!(w_bh, rfftplan, w_b)

    L⁻¹ = (-grid.Krsq .- 2 * f0^2 / (gr * H[2])).^-1
    CUDA.@allowscalar L⁻¹[1,1] = 0.
    L⁻¹ = A(L⁻¹)

    rhs_full_h = @. - (f0/gr) * (∇2J_ψ2_ψ1h + J_ψ2_fh + J_ψ2_ζ2h - J_ψ1_fh - J_ψ1_ζ1h - U1_∂xζ1h + (f0 / H[2]) * w_bh) + ∇2J_ψ2_S32h

    rhs_p2p1_h = @. - (f0/gr) * ∇2J_ψ2_ψ1h
    rhs_p2f_h  = @. - (f0/gr) * J_ψ2_fh
    rhs_p2z2_h = @. - (f0/gr) * J_ψ2_ζ2h
    rhs_p1f_h  = @.   (f0/gr) * J_ψ1_fh
    rhs_p1z1_h = @.   (f0/gr) * J_ψ1_ζ1h
    rhs_U1z1_h = @.   (f0/gr) * U1_∂xζ1h
    rhs_wb_h   = @. - (f0/gr) * (f0 / H[2]) * w_bh
    rhs_S32_h  = ∇2J_ψ2_S32h

    omega_full_h = L⁻¹ .* rhs_full_h
    omega_full = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_full, rfftplan, omega_full_h)

    omega_p2p1_h = L⁻¹ .* rhs_p2p1_h
    omega_p2p1 = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_p2p1, rfftplan, omega_p2p1_h)

    omega_p2f_h = L⁻¹ .* rhs_p2f_h
    omega_p2f = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_p2f, rfftplan, omega_p2f_h)

    omega_p2z2_h = L⁻¹ .* rhs_p2z2_h
    omega_p2z2 = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_p2z2, rfftplan, omega_p2z2_h)

    omega_p1f_h = L⁻¹ .* rhs_p1f_h
    omega_p1f = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_p1f, rfftplan, omega_p1f_h)

    omega_p1z1_h = L⁻¹ .* rhs_p1z1_h
    omega_p1z1 = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_p1z1, rfftplan, omega_p1z1_h)

    omega_U1z1_h = L⁻¹ .* rhs_U1z1_h
    omega_U1z1 = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_U1z1, rfftplan, omega_U1z1_h)

    omega_wb_h = L⁻¹ .* rhs_wb_h
    omega_wb = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_wb, rfftplan, omega_wb_h)

    omega_S32_h = L⁻¹ .* rhs_S32_h
    omega_S32 = deepcopy(vars.u[:,:,1])
    ldiv2D!(omega_S32, rfftplan, omega_S32_h)

    kr = prob.grid.kr
    l = prob.grid.l
    Kr = @. sqrt(kr^2 + l^2)

    krmax = maximum(kr)
    lmax = maximum(abs.(l))
    Kmax = sqrt(krmax^2 + lmax^2)
    Kmin = 0.

    dkr = 2 * pi / Lx
    dl = dkr
    dKr = sqrt(dkr^2 + dl^2)

    K = grid.kr
    K_id = lastindex(K)

    j = argmin(abs.(kr .* Ld .- 2.5))
    # Define high-pass filter matrix
    lpf = ifelse.(Kr .< K[j], Kr ./ Kr, 0 .* Kr)
    lpf[1,1] = 1.0

    lpf = ones(size(ψ1h))

    # lpf[2:end-1, :] .= 0.

    # lpf[:,1] .= 1.0

    τh = 0.5 .* (ψ1h .- ψ2h)

    # Filter the Fourier transformed fields
    τh_lpf = τh .* lpf

    # Inverse transform the filtered fields
    τ_lpf = 0.5 .* (ψ1 .- ψ2) #

    # τ_lpf = A(zeros(Nx, Nx))  #
    # ldiv2D!(τ_lpf, rfftplan, deepcopy(τh_lpf))

    CBC = @. 0.5 * (v1 + v2) * (ψ1 .- ψ2) # τ_lpf

    DBC = @. -0.5 * ζ2 * (ψ1 - ψ2)

    EKE = @. (u1+u2)^2 + (v1+v2)^2

    return CBC, DBC, EKE
end


i=7
j=1
k=2


h0 = (h0s[i], 0.0)
kappa_loc = kappas[j]
mu_loc = mus[k]

global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, (0., 0.), h0)
global topo_PV, eta = define_topo(model_params)
global model_params = redef_mu_kappa_topoPV_h0(model_params, mu_loc, kappa_loc, topo_PV, h0)

beta = β = betas[7]
global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)
a2 = load(data_dir*jld_name_2L(model_params,71.95))
ψ = a2["jld_data"]["psi_yrs_end"]

CBC_beta0, DBC_beta0, EKE_beta0 = calc_CBC_DBC_EKE(ψ)

beta = β = betas[10]
global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, beta)
a2 = load(data_dir*jld_name_2L(model_params,71.95))
ψ = a2["jld_data"]["psi_yrs_end"]

CBC_beta0p75, DBC_beta0p75, EKE_beta0p75 = calc_CBC_DBC_EKE(ψ)

######################################################################################################
######################################################################################################




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

a = load(data_dir*jld_name_2L(model_params,71.95))


###################################################################################
###################################################################################
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html
fig, ax = plt.subplots(3, 3, layout="constrained", figsize=(9.5, 7), height_ratios=[1., 0.005, 1.]) # , width_ratios = [1.2,1.,1.,1.])

ax1=ax[1]; ax2=ax[4]; ax3=ax[7];
ax[2].axis("off");
ax[5].axis("off");
ax[8].axis("off");
ax4=ax[3]; ax5=ax[6]; ax6=ax[9];

# clear the left column for the subfigure:
# for a in axs[:, 1]
#     a.remove()
# end

###########################################################################
############################################################################
# fig, ax = plt.subplots(1,3, figsize=(10,3))
# fig.tight_layout(pad=-2.0)

tsize = 20
fsize = 18
lsize = 12

x = y = collect(range(0, Lx / 2 / pi / Ld, 1024))

dx = dy = x[2]-x[1]

lower_x_lim = 1
upper_x_lim = 1024

lower_y_lim = 1
upper_y_lim = 1024

########################################################################################################

lim1 = maximum(EKE_beta0)

pc1 = ax1.pcolormesh(x, y, EKE_beta0', vmin=-lim1, vmax=lim1, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc1.set_edgecolor("face")

########################################################################################################

lim2 = maximum(abs.(CBC_beta0))

pc2 = ax2.pcolormesh(x, y, CBC_beta0', vmin=-lim2, vmax=lim2, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc2.set_edgecolor("face")

# ax2.contour(x, y, EKE_beta0', colors="black", levels=[0.25 * maximum(EKE_beta0)])

########################################################################################################

lim3 = maximum(abs.(DBC_beta0))

pc3 = ax3.pcolormesh(x, y, DBC_beta0', vmin=-lim3, vmax=lim3, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc3.set_edgecolor("face")

# ax3.contour(x, y, EKE_beta0', colors="black", levels=[0.25 * maximum(EKE_beta0)])

########################################################################################################

ax2.set_yticklabels([])
ax3.set_yticklabels([])


ax1.text(0.025, 0.975, L"\mathbf{(a)} \  \mathrm{EKE}_\mathrm{BT}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax1.transAxes);
ax2.text(0.025, 0.95, L"\mathbf{(b)} \ P ", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax2.transAxes);
ax3.text(0.025, 0.975, L"\mathbf{(c)} \ D_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax3.transAxes);


ax2.text(0.5, 1.115, L"\kappa^* = 0.25, \, \beta^* = 0.0", fontsize=tsize, horizontalalignment="center",verticalalignment="center", transform = ax2.transAxes,
    bbox=Dict("facecolor"=>"none", "edgecolor"=>"black"));

############################################################################
###########################################################################

beta = betas[10]


########################################################################################################

lim1 = maximum(abs.(EKE_beta0p75))

pc4 = ax4.pcolormesh(x, y, EKE_beta0p75', vmin=-lim1, vmax=lim1, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc4.set_edgecolor("face")

########################################################################################################

lim2 = maximum(abs.(CBC_beta0p75))

pc5 = ax5.pcolormesh(x, y, CBC_beta0p75', vmin=-lim2, vmax=lim2, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc5.set_edgecolor("face")

# ax5.contour(x, y, EKE_beta0p75', colors="black", levels=[0.25 * maximum(EKE_beta0p75)])

########################################################################################################

lim3 = maximum(abs.(DBC_beta0p75))

pc6 = ax6.pcolormesh(x, y, DBC_beta0p75', vmin=-lim3, vmax=lim3, cmap = PyPlot.cm.bwr, linewidth=0., rasterized=true)

pc6.set_edgecolor("face")

# ax6.contour(x, y, EKE_beta0p75', colors="black", levels=[0.25 * maximum(EKE_beta0p75)])



########################################################################################################

ax5.set_yticklabels([])
ax6.set_yticklabels([])

ax5.text(0.5, 1.115, L" \kappa^* = 0.25, \, \beta^* = 0.75", fontsize=tsize, horizontalalignment="center",verticalalignment="center", transform = ax5.transAxes,
    bbox=Dict("facecolor"=>"none", "edgecolor"=>"black"));

ax4.text(0.025, 0.975, L"\mathbf{(d)} \ \mathrm{EKE}_\mathrm{BT}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax4.transAxes);
ax5.text(0.025, 0.95, L"\mathbf{(e)} \ P ", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax5.transAxes);
ax6.text(0.025, 0.975, L"\mathbf{(f)} \ D_\mathrm{BC}", fontsize=fsize, horizontalalignment="left",verticalalignment="top", transform = ax6.transAxes);

# ax1.grid()
# ax2.grid()
# ax3.grid()





############################################################################
###########################################################################
# make the subfigure in the empty gridspec slots:



# for axn in ax[4:end]
#     axn.set_yticklabels([])
# end
# for axn in [ax[1:2]; ax[4:5]; ax[7:8]]
#     axn.set_xticklabels([])
# end

for axn in ax
    axn.tick_params(labelsize=lsize)
    axn.set_xlabel(L"x / 2 \pi \lambda ", fontsize=fsize)

    axn.set_xlim(x[lower_x_lim], x[upper_x_lim])
    axn.set_ylim(y[lower_y_lim], y[upper_y_lim])

end

ax1.set_ylabel(L"y / 2 \pi \lambda ", fontsize=fsize)
ax4.set_ylabel(L"y / 2 \pi \lambda ", fontsize=fsize)

# subfig.set_facecolor('0.75')
# for ax in axsLeft:
#     pc = example_plot(ax)
# subfig.suptitle('Left plots', fontsize='x-large')
# subfig.colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

# fig.suptitle('Figure suptitle', fontsize='xx-large')
# plt.show()


savefig("./JFM_BC_nrg_figs/2D_fields.pdf",bbox_inches="tight")


















