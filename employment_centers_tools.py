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
import matplotlib.pyplot as plt
from itertools import izip,count
from pyGDsandbox.geo_tools import clip_shp
from pyGDsandbox.dataIO import df2dbf
from pysal.contrib.viz import mapping as maps
from matplotlib.colors import colorConverter as cc

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

class RLabel:
    """Takes 'all' and obtains a pseudo p-value for statistical difference
    of the mean between the two groups in 'all'.  Allows for testing against
    the universe of observations versus against the remaining observations.
    Arguments:
        *  all=[[values_in_group1],[values_in_group2]] 
        *  useAll = When True test group1 against (group1+group2); when False
                    test group1 against group2
    Attributes:
        * mean0
        * mean1
        * permutations
        * diff=difference of means of observed groups
        * diffs=list of differences of means for simulated groups
        * p_sim
        """
    def __init__(self,all,permutations=99999,useAll=False):
        allT=all[0]+all[1]
        self.permutations=permutations
        if useAll:
            self.mean0,self.mean1=np.mean(all[0]),np.mean(allT)
        else:
            self.mean0,self.mean1=np.mean(all[0]),np.mean(all[1])
        self.diff=self.mean0-self.mean1
        self.absDiff=np.abs(self.diff)
        sep=len(all[0])
        diffs=[self.__calc(allT,sep,useAll) for i in xrange(permutations)]
        self.diffs=diffs
        self.p_sim=(sum(diffs >= self.absDiff)+1.)/(permutations+1.)

    def __calc(self,allT,sep,useAll=False):
        np.random.shuffle(allT)
        if useAll:
            diff = np.abs(np.mean(allT[:sep])-self.mean1)
        else:    
            diff = np.abs(np.mean(allT[:sep])-np.mean(allT[sep:]))
        return diff

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

def evol_tab(db):
    '''
    Build table of evolution. Counts how many MSAs there are in every possible
    combination for the three periods in time (1990, 2000, 2010)
    ...

    Arguments
    ---------
    db      : DataFrame
              Tract table with at least MSA, year and center identifiers as
              columns

    Returns
    -------
    tab     : DataFrame
              List with MSA counts indexed on the three types of MSAs
              (no_centers, monocentric, polycentric) across the three years
    '''
    g = db.groupby(['msa', 'year']).apply(\
            lambda x: x.groupby('center_id').ngroups)
    simp = g.apply(_monopoly)
    tab = simp.unstack().groupby([1990, 2000, 2010]).size()
    return tab

def _monopoly(c):
    if c == 0:
        return 'empty'
    elif c == 1:
        return 'monocentric'
    else:
        return 'polycentric'

q_names = {0: 'Insignificant', 1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
q_mapper = {0: cc.to_rgba('0.3'), 1: (0.75, 0, 0, 1), \
        2: (1.0, 0.7529411764705882, 0.8, 1), \
        3: cc.to_rgba('blue'), \
        4: (0, 0.8, 1, 1)}

def plot_lisa(lisa, st, msa, outfile=None, thr=0.05, title=''):
    '''
    Plot LISA results for MSAs on background map of US states

    NOTE: shapefiles hardcoded linked to paths inside the function
    ...

    Arguments
    ---------
    lisa    : Moran_Local
              LISA object from PySAL
    st      : str
              Path to states shape
    msa     : str
              Path to MSA points shape
    outfile : str
              [Optional] Path to png to be written
    thr     : float
              [Optional] Significance value to identify clusters
    title   : str
              [Optional] Title for the figure
    title   : str
    Returns
    -------
    None
    '''
    sig = (lisa.p_sim < thr) * 1
    vals = pd.Series(lisa.q * sig)

    states = ps.open(st)
    pts = ps.open(msa)

    fig = plt.figure(figsize=(9, 5))

    base = maps.map_poly_shp(states)
    base.set_facecolor('0.85')
    base.set_linewidth(0.75)
    base.set_edgecolor('0.95')

    msas = pd.np.array([pt for pt in ps.open(msa)])
    sizes = vals.apply(lambda x: 4 if x==0 else 50)
    colors = vals.map(q_mapper)
    colors = pd.np.array(list(colors))
    pts = []
    for clas in q_mapper:
        i = vals[vals==clas].index
        p = plt.scatter(msas[i, 0], msas[i, 1], s=sizes[i], \
                c=colors[i, :], label=q_names[clas])
        p.set_linewidth(0)
        pts.append(p)
    plt.legend(loc=3, ncol=2, fontsize=14, scatterpoints=1, frameon=False)

    ax = maps.setup_ax([base] + pts)
    #ax = maps.setup_ax(pts)
    fig.add_axes(ax)
    if title:
        plt.title(title)
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    return None

def load_soc_ec(link):
    msa = 'm' + link.split('/')[-1].strip('m').strip('.csv')
    db = pd.read_csv(link, index_col=0).rename(_guess)
    db['msa'] = msa
    return db

def _guess(id):
    id = str(id)
    if len(id) == 11: # 48 999 999 999
        return 'G' + str(id)
    if len(id) == 10: # 06 999 999 999
        return 'G0' + str(id)
    if len(id) == 13: # 48 999 999 999 00
        return 'G' + str(id)
    if len(id) == 12: # 06 999 999 999 00
        return 'G0' + str(id)

def do_rl(msas, years, perms=99):
    g90_00 = msas.groupby(years)
    out = []
    for g in g90_00:
        id, g = g
        sub = []
        for var in g.drop([1990, 2000, 2010], axis=1):
            g1 = g[var]
            rest = msas.ix[msas.index - g1.index, var]
            all = [list(g1.values), list(rest.values)]
            r = RLabel(all, permutations=perms, useAll=True)
            cell = str(r.mean0) + _sign(r) + _signify(r.p_sim)
            s = pd.Series(cell, index=[var])
            sub.append(s)
        sub = pd.concat(sub)
        sub.name = id
        out.append(sub)
    out = pd.concat(out, axis=1).T
    out.index = pd.MultiIndex.from_tuples(out.index, \
            names=years)
    return out

def _sign(r):
    if (r.mean0 - r.mean1) > 0 and r.p_sim < 0.1:
        return '+'
    elif (r.mean0 - r.mean1) <= 0 and r.p_sim < 0.1:
        return '-'
    else:
        return ''

def _signify(p):
    if p < 0.01:
        return '***'
    elif 0.01 <= p < 0.05:
        return '**'
    elif 0.05 <= p < 0.1:
        return '*'
    elif p >= 0.1:
        return ''
