"""Dense wall sampling, identical polygon and independent cold starts."""
import numpy as np
import gsfit_rs
_CONFIGURED=False

def configure():
    global _CONFIGURED
    if _CONFIGURED:return
    real=gsfit_rs.Plasma
    def plasma(*args, **kwargs):
        if kwargs or len(args)!=18:raise ValueError("unexpected Plasma constructor")
        args=list(args)
        points=np.column_stack([args[7],args[8]])
        if not np.array_equal(points[0],points[-1]):points=np.vstack([points,points[0]])
        dense=[]
        for left,right in zip(points[:-1],points[1:]):
            n=max(1,int(np.ceil(np.linalg.norm(right-left)/.03)))
            dense.extend(left+(right-left)*np.arange(n)[:,None]/n)
        dense=np.asarray(dense);args[7]=dense[:,0].copy();args[8]=dense[:,1].copy()
        return real(*args)
    gsfit_rs.Plasma=plasma
    _CONFIGURED=True
