import sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
name=sys.argv[1]; z=np.load(f'out/{name}_grid.npz'); gs,gd,gn,o,B=z['gs'],z['gd'],z['gn'],z['o'],z['B']
comp=pd.read_csv('out/composites.csv'); x0,x1={'FarWest':(369500,370500),'West':(370500,372050),'East':(372050,373250),'FarEast':(373250,374300)}[name]
c=comp[(comp.X>=x0)&(comp.X<x1)&comp.form.notna()]
if 'z0' in z.files:
    from unfold import Unfold; L=Unfold(comp)(c[['X','Y','Z']].to_numpy()); L[:,1]-=float(z['z0'])
else: L=(c[['X','Y','Z']].to_numpy()-o)@B.T
fig,ax=plt.subplots(2,1,figsize=(11,8.2),constrained_layout=True)
ext=[gs[0],gs[-1],gd[0],gd[-1]]
for a,(k,lab,cm) in zip(ax,[('MS','Massive sulphide true thickness (m)','Reds'),('OH','Ore horizon true thickness (m)','Blues')]):
    t=(z[k]>=0).sum(2)*2.5; t=np.where(t>0,t,np.nan)
    im=a.imshow(t.T,origin='lower',extent=ext,cmap=cm,aspect='equal',vmin=0,vmax=np.nanpercentile(t,98))
    a.scatter(L[:,0],L[:,1],s=0.15,c='k',alpha=.25,lw=0); a.set_title(f'{name}: {lab}, unfolded along the ore horizon'); a.set_xlabel('along the ore-horizon trace (m)'); a.set_ylabel('elevation relative to mean (m)')
    fig.colorbar(im,ax=a,shrink=.8)
fig.savefig(f'out/{name}_qc.png',dpi=100)
