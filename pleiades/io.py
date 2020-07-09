import matplotlib.pyplot as plt
from pleiades.analysis import *

class EQDSK(dict):
    def __init__(self,fname=None):
        if fname is not None:
            self.read_eqdsk(fname)
        else:
            pass

    def write_eqdsk(self,fname):
        with open(fname,"w") as f:
            pass

    def read_eqdsk(self,fname):
        with open(fname,"r") as f:
            pass

#class MirrorEQDSK(EQDSK):
#    def __init__(self,fname=None):
#        self["
#
#
#        ## line 1 -- write grid information: cursign, nnr, nnz, nnv
#        f.write(title+"".join("{:4d}".format(xi) for xi in [int(cursign),nnr,nnz,nnv]))
#        ## line 2 -- write rbox,zbox,0,0,0
#        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [rbox,zbox,0,0,0]))
#        ## line 3 -- write 0,0,0,psi_lim,0
#        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,psi_lim,0]))
#        ## line 4 -- write total toroidal current,0,0,0,0
#        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [tot_cur,0,0,0,0]))
#        ## line 5 -- write 0,0,0,0,0
#        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,0,0]))


def write_eqdsk(Rho,Z,psi,plas_currents,fname,title):
    title='PLEIADES '+title
    if len(title) >= 26:
        title = title[0:26]
    title = title.ljust(26) + "cursign,nnr,nnz,nnv = "
    nnr,nnz,nnv = int(len(Rho[0,:])),int(len(Z[:,0])),101
    rbox,zbox = np.amax(Rho)-np.amin(Rho),np.amax(Z)-np.amin(Z)
    tot_cur = np.sum(plas_currents)
    cursign = np.sign(tot_cur)
    blank = np.zeros(nnv)
    limit_pairs,vessel_pairs = 100,100
    rho0_idx = np.abs(Rho[0,:]).argmin()
    #### Plotting current and flux lines from plasma currents 
    [psi_lim] = locs_to_vals(Rho,Z,psi,[(.6,0)])
    psi_ves = psi_lim*1.02
    psi_levels = tuple(sorted([psi_lim,psi_ves]))
    fig,ax = plt.subplots()
    cf = ax.contour(Rho,Z,psi,psi_levels,colors='k',zorder=1)
    ## get contour for psi_lim boundary
    flpoints = get_fieldlines(cf,psi_lim,start_coord=(.05,.5),end_coord=(.05,-.5))
    r,z = flpoints[:,0],flpoints[:,1]
    z = np.array(list(z)+[-z[0]])
    z = z[::-1]
    r = np.array(list(r)+[r[0]])
    r = r[::-1]
    flpoints = np.vstack((z,r)).T
#    ax.plot(flpoints[:,0],flpoints[:,1],"bo")
#    ax.plot(flpoints[0,0],flpoints[0,1],"go")
#    ax.plot(flpoints[-1,0],flpoints[-1,1],"ro")
#    plt.show()
    fl_dist = get_fieldline_distance(flpoints)
    spl = UnivariateSpline(z,r,k=1,s=0)
    fl_spl = UnivariateSpline(fl_dist,z,k=1,s=0)
    uniform_s = np.linspace(fl_dist[0],fl_dist[-1],100)
    zlimit = fl_spl(uniform_s)
    rlimit = spl(zlimit)
#    ax.plot(r,z,"bo")
#    ax.plot(rlimit,zlimit,"ro")
    ## get contour for psi_ves boundary
    flpoints = get_fieldlines(cf,psi_ves,start_coord=(.05,.5),end_coord=(.05,-.5))
    r,z = flpoints[:,0],flpoints[:,1]
    z = np.array(list(z)+[-z[0]])
    z = z[::-1]
    r = np.array(list(r)+[r[0]])
    r = r[::-1]
    flpoints = np.vstack((z,r)).T
    fl_dist = get_fieldline_distance(flpoints)
    spl = UnivariateSpline(z,r,k=1,s=0)
    fl_spl = UnivariateSpline(fl_dist,z,k=1,s=0)
    uniform_s = np.linspace(fl_dist[0],fl_dist[-1],100)
    zves = fl_spl(uniform_s)
    rves = spl(zves)
#    ax.plot(r,z,"yo")
#    ax.plot(rves,zves,"go")
#    plt.show()
    plt.close()

    lim_ves_pairs = [loc for pair in zip(rlimit,zlimit) for loc in pair]+[loc for pair in zip(rves,zves) for loc in pair]

    with open(fname,"w") as f:
        ## line 1 -- write grid information: cursign, nnr, nnz, nnv
        f.write(title+"".join("{:4d}".format(xi) for xi in [int(cursign),nnr,nnz,nnv]))
        ## line 2 -- write rbox,zbox,0,0,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [rbox,zbox,0,0,0]))
        ## line 3 -- write 0,0,0,psi_lim,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,psi_lim,0]))
        ## line 4 -- write total toroidal current,0,0,0,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [tot_cur,0,0,0,0]))
        ## line 5 -- write 0,0,0,0,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,0,0]))
        ## line 6 -- write list of toroidal flux for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 7 -- write list of pressure for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 8 -- write list of (RBphi)' for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 9 -- write list of P' for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 10 -- write flattened list of psi values on whole grid (NOT ZERO)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(psi.flatten())))
        ## line 11 -- write list of q for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 12 -- write number of coordinate pairs for limit surface and vessel surface
        f.write("\n"+"".join("{0:5d}{1:5d}".format(limit_pairs,vessel_pairs)))
        ## line 13 -- write list of R,Z pairs for limiter surface, then vessel surface
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(lim_ves_pairs)))


def write_eqdsk_fromdict(eq_dict,fname):
    title = eq_dict["title"]
    cursign,nnr,nnz,nnv = eq_dict["cursign"], eq_dict["nnr"], eq_dict["nnz"], eq_dict["nnv"]
    rbox,zbox = eq_dict["rbox"],eq_dict["zbox"]
    psi_lim = eq_dict["psi_lim"]
    Ip = eq_dict["Ip"]
    p_flux = eq_dict["p_flux"]
    tor_flux = eq_dict["tor_flux"] 
    rbphi_flux = eq_dict["rbphi_flux"]
    pprime_flux = eq_dict["pprime_flux"]
    psi = eq_dict["psi"]
    q_flux = eq_dict["q_flux"]
    nlim_pairs = eq_dict["nlim_pairs"]
    nves_pairs = eq_dict["nves_pairs"]
    lim_pairs = eq_dict["lim_pairs"]
    ves_pairs = eq_dict["ves_pairs"]
    lim_ves_pairs = [loc for pair in lim_pairs for loc in pair] + [loc for pair in ves_pairs for loc in pair]
    with open(fname,"w") as fh:
        ## line 1 -- write grid information: cursign, nnr, nnz, nnv
        fh.write(title+"".join("{:4d}".format(xi) for xi in [int(cursign),nnr,nnz,nnv]))
        ## line 2 -- write rbox,zbox,0,0,0
        fh.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [rbox,zbox,0,0,0]))
        ## line 3 -- write 0,0,0,psi_lim,0
        fh.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,psi_lim,0]))
        ## line 4 -- write total toroidal current,0,0,0,0
        fh.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [Ip,0,0,0,0]))
        ## line 5 -- write 0,0,0,0,0
        fh.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,0,0]))
        ## line 6 -- write list of toroidal flux for each flux surface (zeros)
        fh.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(tor_flux)))
        ## line 7 -- write list of pressure for each flux surface (zeros)
        fh.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(p_flux)))
        ## line 8 -- write list of (RBphi)' for each flux surface (zeros)
        fh.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(rbphi_flux)))
        ## line 9 -- write list of P' for each flux surface (zeros)
        fh.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(pprime_flux)))
        ## line 10 -- write flattened list of psi values on whole grid (NOT ZERO)
        fh.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(psi.flatten())))
        ## line 11 -- write list of q for each flux surface (zeros)
        fh.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(q_flux)))
        ## line 12 -- write number of coordinate pairs for limit surface and vessel surface
        fh.write("\n"+"".join("{0:5d}{1:5d}".format(nlim_pairs,nves_pairs)))
        ## line 13 -- write list of R,Z pairs for limiter surface, then vessel surface
        fh.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(lim_ves_pairs)))

def read_eqdsk(filename):
    """
    line 1 -- read grid information: title cursign, nnr, nnz, nnv
    line 2 -- read rbox,zbox,0,0,0
    line 3 -- read 0,0,0,psi_lim,0
    line 4 -- read total toroidal current,0,0,0,0
    line 5 -- read 0,0,0,0,0
    line 6 -- read list of toroidal flux for each flux surface (zeros)
    line 7 -- read list of pressure for each flux surface (zeros)
    line 8 -- read list of (RBphi)' for each flux surface (zeros)
    line 9 -- read list of P' for each flux surface (zeros)
    line 10 -- read flattened list of psi values on whole grid (NOT ZERO)
    line 11 -- read list of q for each flux surface (zeros)
    line 12 -- read number of coordinate pairs for limit surface and vessel surface
    line 13 -- read list of R,Z pairs for limiter surface, then vessel surface
    """
    eq_dict = {}
    with open(filename,"r") as f:
        lines = f.readlines()
        line1 = lines[0]
        eq_dict["title"] = line1[0:48]
        line1rem = line1[48:]
        eq_dict["cursign"] = int(line1rem.split()[-4])
        eq_dict["nnr"] = int(line1rem.split()[-3])
        eq_dict["nnz"] = int(line1rem.split()[-2])
        eq_dict["nnv"] = int(line1rem.split()[-1])
        line2 = lines[1].split()
        eq_dict["rbox"] = float(line2[-5])
        eq_dict["zbox"] = float(line2[-4])
        line3 = lines[2].split()
        eq_dict["psi_lim"] = float(line3[-2])
        line4 = lines[3].split()
        eq_dict["Ip"] = float(line4[-5])
        line5 = lines[4].split()
        fs_lines = int(np.ceil(eq_dict["nnv"]/5.0))
        head = [line.strip().split() for line in lines[5:5+fs_lines]]
        tor_flux = np.array([float(num) for line in head for num in line])
        eq_dict["tor_flux"] = tor_flux
        head = [line.strip().split() for line in lines[5+fs_lines:5+2*fs_lines]]
        p_flux = np.array([float(num) for line in head for num in line])
        eq_dict["p_flux"] = p_flux
        head = [line.strip().split() for line in lines[5+2*fs_lines:5+3*fs_lines]]
        rbphi_flux = np.array([float(num) for line in head for num in line])
        eq_dict["rbphi_flux"] = rbphi_flux
        head = [line.strip().split() for line in lines[5+3*fs_lines:5+4*fs_lines]]
        pprime_flux = np.array([float(num) for line in head for num in line])
        eq_dict["pprime_flux"] = pprime_flux
        # Read psi on whole grid, nnr x nnz
        nnr,nnz = eq_dict["nnr"],eq_dict["nnz"]
        rz_pts = nnr*nnz
        l0 = 5+4*fs_lines
        psi_lines = int(np.ceil(rz_pts/5.0))
        head = [line.strip().split() for line in lines[l0:l0+psi_lines]]
        psi = np.array([float(num) for line in head for num in line])
        eq_dict["psi"] = psi.reshape((nnz,nnr))
        rbox,zbox = eq_dict["rbox"],eq_dict["zbox"]
        R,Z = np.meshgrid(np.linspace(0,rbox,nnr),np.linspace(-zbox/2,zbox/2,nnz))
        eq_dict["R"] = R
        eq_dict["Z"] = Z
        head = [line.strip().split() for line in lines[l0+psi_lines:l0+psi_lines+fs_lines]]
        q_flux = np.array([float(num) for line in head for num in line])
        eq_dict["q_flux"] = q_flux
        nlim_pairs, nves_pairs = [int(x) for x in lines[l0+psi_lines+fs_lines].strip().split()]
        eq_dict["nlim_pairs"] = nlim_pairs
        eq_dict["nves_pairs"] = nves_pairs
        pair_lines = int(np.ceil((nlim_pairs + nves_pairs)*2.0/5.0))
        rest = [line.rstrip() for line in lines[l0+psi_lines+fs_lines+1:]]
        rest = [line[i:i+16] for line in rest for i in range(0,len(line),16)]
        pairs = np.array([float(num.strip()) for num in rest])
        lim_pairs = np.array(list(zip(pairs[0:2*nlim_pairs:2],pairs[1:2*nlim_pairs:2])))
        ves_pairs = np.array(list(zip(pairs[2*nlim_pairs::2],pairs[2*nlim_pairs+1::2])))
        eq_dict["lim_pairs"] = lim_pairs
        eq_dict["ves_pairs"] = ves_pairs

        return eq_dict


