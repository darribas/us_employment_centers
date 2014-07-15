'''
Employment center identification tools
...

Copyright (c) 2014, Daniel Arribas-Bel
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* The name of Daniel Arribas-Bel may not be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import os
import numpy as np
import pandas as pd
import pysal as ps
import multiprocessing as mp
from itertools import izip,count
from pyGDsandbox.geo_tools import clip_shp
from pyGDsandbox.dataIO import df2dbf

class CentFinder():
    """
    Identify employment centers out of LISA results
    ...

    Arguments
    ---------
    lisas       : ndarray
                  Input data with the following structure:

                    lisas[i,:] = (Ii, z_sim, EI_sim, seI_sim, p_sim, q)

    w           : W 
                  Spatial weights

    threshold   : float
                  Significance level

    verbose     : Boolean
                  False by default

    Attributes
    ----------
    lisas       : ndarray
                  Original input

    ps          : ndarray
                  List of lenght n where every element gets its p_sim if it's
                  HH/HL, 1.1 otherwise

    classes     : ndarray
                  List of lenght n where every element gets 1 if HH, 2 if HL, 0 otherwise

    sClus       : dict
                  Mapping of core tract ID (key) to IDs of tracts in the
                  center
    """
    def __init__(self,lisas,w,threshold,verbose=False):
        self.verbose=verbose
        self.threshold=threshold
        results=lisas
        self.lisas=lisas
        self.n=lisas.shape[0]
        self.w=w
        classes=np.zeros(self.n)
        ps=np.ones(self.n)+0.1
        for i in range(self.n):
            if results[i,5]==1.:
                ps[i]=results[i,4]
                classes[i]=1.
            if results[i,5]==4.:
                ps[i]=results[i,4]
                classes[i]=2.
        self.classes=classes
        mp=min(izip(ps,count())) # (min,map)
        self.ps=ps
        if mp[0]>self.threshold:
            cores=[]
            sClus={}
            self.sClus=sClus
            self.cores=cores
        else:
            sClus={} #sClus[candidate]=set([cand])
            for i in range(self.n):
                if ps[i]<=self.threshold: # si candidato
                    sClus[w.id_order[i]]=set([w.id_order[i]])

            # Check contiguity of the clusters
            several=1
            if len(sClus)<2:
                several=0
            if several:
                flag=1
                while flag:
                    cores=sClus.keys()#tract_id's
                    for indi,cl_main in enumerate(cores):
                        if verbose:
                            print '\nMAIN: ',cl_main,'\n'
                        trash=[]
                        for cl_an in cores:
                            if cl_main != cl_an: #if not the same cluster
                                if verbose:
                                    print 'analyzing ',cl_an
                                for tract in sClus[cl_main]:
                                    sn=set(self.w.neighbors[tract])
                                    if sn.intersection(sClus[cl_an]):
                                        sClus[cl_main]=sClus[cl_main].union(sClus[cl_an])
                                        trash.append(cl_an)
                                        if verbose:
                                            print cl_an,' and ',cl_main,' neigh\n'
                                        break
                        if trash:
                            for i in trash:
                                del sClus[i]
                            break
                        elif indi==len(cores)-1:
                            flag=0
            sClusNew={}
            newCores=[]
            for i in sClus:
                minp=('ph',1)
                for j in sClus[i]:
                    if results[w.id_order.index(j),4]<minp[1]:
                        minp=(j,results[w.id_order.index(j),4])
                sClusNew[minp[0]]=sClus[i]
                newCores.append(minp)
            self.sClus=sClusNew
            self.cores=newCores

def act_on_msa(empShpOut_paths, thr=0.1, permutations=9999):
    '''
    Perform operations required at the MSA level

    NOTE: besides returning `msa`, the method creates a shapefile and a .gal
    file for the MSA (if not present in `out_path`) and a shapefile with
    centers in `out_path`
    ...

    Arguments
    ---------
    msaEmpShp_path  : tuple
                      Parameters, including:

                      * emp: DataFrame with MSA data
                      * shp_path: None/str to shp with all tracts
                      * out_path: str to out folder
    thr             : float
                      [Optional, default to 0.1] Significance level to consider center candidates
    permutations    : int
                      [Optional, default to 9999] Number of permutations to
                      obtain pseudo-significance values

    Returns
    -------
    msa             : DataFrame
                      Table with output information for tracts in msa. This
                      includes:

                      * dens_eb
                      * lisa_i
                      * lisa_p_sim
                      * center_id
    '''
    emp, shp_link, out_link = empShpOut_paths
    msa = emp['msa'].min()
    # get shape and W
    msa_shp_link = out_link + msa + '.shp'
    msa_gal_link = msa_shp_link.replace('.shp', '_queen.gal')
    try:
        fo = ps.open(msa_shp_link)
        fo.close()
    except:
        _ = clip_shp(shp_link, "GISJOIN", list(emp['GISJOIN'].values), \
                msa_shp_link)
    try:
        w = ps.open(msa_gal_link).read()
    except:
        w = ps.queen_from_shapefile(msa_shp_link, "GISJOIN")
        fo = ps.open(msa_gal_link, 'w')
        fo.write(w)
        fo.close()
    print w.weights.keys()[:5]
    w.id_order = w.id_order
    print w.weights.keys()[:5]
    w.transform = 'R'
    print w.weights.keys()[:5]
    emp = emp.set_index('GISJOIN').reindex(ps.open(\
            msa_shp_link.replace('.shp', '.dbf'))\
            .by_col('GISJOIN'))\
            .fillna(0)
    # get EB rate
    print w.weights.keys()[:5]
    eb = ps.esda.smoothing.Spatial_Empirical_Bayes(\
            emp['emp'].values, emp['Shape_area'].values, w)
    emp['dens_eb'] = eb.r
    emp['dens_eb'] = emp['dens_eb'].fillna(0) #Avoid sliver problem
    # LISA
    lisa = ps.Moran_Local(emp['dens_eb'].values, w, permutations=permutations)
    lisa_pack = pd.DataFrame({'Is': lisa.Is, 'z_sim': lisa.z_sim, \
            'EI_sim': lisa.EI_sim, 'seI_sim': lisa.seI_sim, \
            'p_sim': lisa.p_sim, 'q': lisa.q})
    lisa_pack = lisa_pack[['Is', 'z_sim', 'EI_sim', 'seI_sim', 'p_sim', 'q']]
    emp['lisa_i'] = lisa.Is
    emp['lisa_p_sim'] = lisa.p_sim
    emp['q'] = lisa.q
    emp['q'] = emp['q'].map({1: 'HH', 2: 'LH', 3: 'LL', 4: 'LH'})
    # Center identification
    w.transform = 'O'
    c = CentFinder(lisa_pack.values, w, thr)
    emp['center_id'] = None
    for core in c.sClus:
        members = list(c.sClus[core])
        emp.ix[members, 'center_id'] = core
    # Write out results
    if c.sClus:
        cent_shp_link = out_link + msa + '_cent.shp'
        ids_in_cent = list(emp[emp['center_id'].notnull()].index.values)
        _ = clip_shp(msa_shp_link, "GISJOIN", ids_in_cent, cent_shp_link)
        _ = df2dbf(emp.reindex(ps.open(cent_shp_link\
                .replace('.shp', '.dbf')).by_col('GISJOIN')),
                cent_shp_link.replace('.shp', '.dbf'))
    emp.index.name = "GISJOIN"
    emp.to_csv(out_link + msa + '.csv')
    return emp

def load_msa_data(link, y90=False):
    """
    Load legacy 1990 and 2000 data
    ...

    Arguments
    ---------
    link    : str
              Path to original data
    y90     : boolean
              Flag for 1990 data. If False (default), it assumes the length of
              a GISJOIN id is 14 with `G` included; if True, it assumes a
              length of 12.

    Returns
    -------
    db      : DataFrame
              Table indexed to GISJOIN with `emp`, `Shape_area` and `msa` as
              columns
    """
    def _guess90(id):
        id = str(id)
        if len(id) == 11: # 48 999 999 999
            return 'G' + str(id)
        if len(id) == 10: # 06 999 999 999
            return 'G0' + str(id)
        if len(id) == 13: # 48 999 999 999 00
            return 'G' + str(id)
        if len(id) == 12: # 06 999 999 999 00
            return 'G0' + str(id)

    db = pd.read_csv(link, index_col=0)[['emp', 'area']]
    db['area'] = db['area'] * 9.2903e-8# Sq. foot to Sq. Km
    db['msa'] = 'm' + link.strip('.csv').split('/')[-1]
    if y90:
        db = db.rename(lambda x: _guess90(x))\
                .rename(columns={'area': 'Shape_area'})
    else:
        db = db.rename(lambda x: 'G' + str(x).zfill(13))\
                .rename(columns={'area': 'Shape_area'})
    return db

def msafy(cty, cty2msa):
    '''
    Helper function to assign MSA to a county
    ...

    Arguments
    ---------
    cty         : str
                  County to assign a MSA
    cty2msa     : dict
                  Mapping of counties to MSAs

    Returns
    -------
    MSA/None
    '''
    try:
        return cty2msa[cty]
    except:
        return None

if __name__ == "__main__":

    #pool = mp.Pool(mp.cpu_count())
    seed = np.random.seed(1234)

    cent_out90 = '/home/dani/AAA/LargeData/T-CentersData/centers/oc0p1/noControl/nc1990rerun/'
    cent_out90 = '/Users/dani/Desktop/test90/'
    cent_out00 = '/home/dani/AAA/LargeData/T-CentersData/centers/oc0p1/noControl/nc2000rerun/'
    cent_out10 = '/home/dani/AAA/LargeData/T-CentersData/centers/oc0p1/noControl/nc2010/'

    cent_in90 = '/Users/dani/AAA/LargeData/T-CentersData/shapes/msaTracts1990polygonsSP/'

    run90 = True
    if run90:
        empF90 = '/Users/dani/AAA/LargeData/T-CentersData/attributes/3-empDen/empDen1990/'
        emp90 = pd.concat([load_msa_data(empF90+f, y90=True) for f in os.listdir(empF90)])
        emp90['dens_raw'] = (emp90['emp'] * 1.) / emp90['Shape_area']
        emp90['GISJOIN'] = emp90.index
        '''
        pars = [(emp90[emp90['msa']==msa], None, cent_out90) \
                for msa in emp90['msa'].unique()]
        '''
        pars = [(emp90[emp90['msa']==msa], cent_in90+msa[1:]+'.shp', cent_out90) \
                for msa in emp90['msa'].unique()]

        out = map(act_on_msa, pars[:1])

