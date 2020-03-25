from numpy import (pi, linspace, meshgrid, sin, cos, sqrt, sum, array, ones,
                   zeros, hstack, vstack, sign, mod, isfinite, ceil, isclose)
from scipy.special import ellipk, ellipe
from multiprocessing import Process, Queue, cpu_count
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from numbers import Number
from warnings import warn, simplefilter
import pickle


class Current(object):
    """Represents an axisymmetric ring of toroidal current.

    Parameters
    ----------
    loc : tuple, optional
        The (R, Z) location of the current centroid in meters. Defaults to None
    current : float, optional
        The current in the ring in amps, defaults to 1 amp.

    Attributes
    ----------
    loc : tuple, optional
        The (R, Z) location of the current centroid in meters. Defaults to None
    current : float, optional
        The current in the ring in amps, defaults to 1 amp.
    marker : str
        A matplotlib marker string forplotting current direction "x" for "into
        the page", etc.

    """
    def __init__(self, loc=None, current=1.0):
        self.current = current
        self.loc = loc

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc):
        r, z = loc
        if r <= 0:
            warn('Current R location in left half plane, setting it to zero',
                 UserWarning)
            r = 0
            self.current = 0
        else:
            self.current = self._current
        self._loc = (float(r), float(z))

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, newcurrent):
        #Set current value in Amps + for ZxR direction - for RxZ
        self._current = float(newcurrent)
        if self._current == 0:
            self.marker = ''
        elif self._current < 0:
            self.marker = 'ko'
        else:
            self.marker = 'kx'

    def plot(self, ax):
        """Plot current locations with markers for +/-

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axis on which to plot the current location
        """
        r0, z0 = self._loc
        ax.plot(r0, z0, self.marker)

    def to_dict(self):
        """Represent Current object as a dictionary"""
        cls = str(self.__class__).split("'")[1]
        return {"cls": cls, "loc": self._loc, "current": self._current}

    @classmethod
    def from_dict(cls, cls_dict):
        """Create Current instance from a dictionary
        
        Parameters
        ----------
        cls_dict : dict
            The dictionary from which to construct a Current.
        """
        cls = cls_dict.pop("cls", None)
        return cls(**cls_dict)


class CurrentGroup(object):
    """ Grouping of Current objects that have the same current value

    Parameters
    ----------
    rz_pts : iterable, optional
        Nx2 iterable representing R,Z current centroids. Defaults to None
    current : float, optional
        The current in all the current ring in amps, defaults to 1 amp.
    kwargs : matplotlib patch keyword arguments

    Attributes
    ----------
    current : float
        The current in the CurrentGroup in amps.
    obj_list : list
        The list of Current objects that comprise the CurrentGroup
    rzdir : np.array
        An Nx3 array whos rows are rzdir[i, :] = rloc, zloc, current which
        describe the current location and current value for each current in the
        CurrentGroup
    patch : matplotlib.patches.Patch object
        The patch object representing the CurrentGroup for plotting
    patchkwargs : dict
        The keyword arguments used for the patch attribute
    """

    def __init__(self, rz_pts=None, current=1.0, **kwargs):
        CurrentGroup.reset(self, rz_pts=rz_pts, current=current, **kwargs)

    def reset(self, **kwargs):
        rz_pts = array(kwargs.pop("rz_pts", None))
        if rz_pts is None:
            raise ValueError("rz_pts must not be None")
        current = float(kwargs.pop("current", 1.0))
        self.patchcls = kwargs.pop("patchcls", None)
        self.patchargs_dict = kwargs.pop("patchargs_dict", {})
        n, d = rz_pts.shape
        if not (d == 2 and n >= 1):
            raise ValueError("rz_pts shape: {0} is invalid, must be Nx2".format(rz_pts.shape))
        self._current = current
        self._obj_list = [Current(loc=(r, z), current=current) for r, z in rz_pts]
        self.patchkwargs = {"fc": "w", "ec": "k", "zorder": 3}
        self.patchkwargs.update(kwargs)
        self.update_patch()

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, new_current):
        self._current = new_current
        for c_obj in self._obj_list:
            c_obj.current = new_current

    @property
    def obj_list(self):
        return self._obj_list

    @obj_list.setter
    def obj_list(self, new_obj_list):
        if not all([type(c_obj) == Current for c_obj in new_obj_list]):
            raise TypeError("All objects in obj_list must be of type core.Current")
        self._obj_list = new_obj_list

    @property
    def rzdir(self):
        return array([c_obj.loc + (1,) for c_obj in self._obj_list], dtype="float32")

    @property
    def patch(self):
        return self._patch

    def translate(self, dr, dz):
        """Translate the current group by the vector (dr, dz)

        Parameters
        ----------
        dr : float
            The displacement in the R direction for the translation
        dz : float
            The displacement in the Z direction for the translation
        """
        for c_obj in self._obj_list:
            r, z = c_obj.loc
            c_obj.loc = r + dr, z + dz
        self.update_patch()

    def rotate(self, r0, z0, angle):
        """Rotate the current group by a given angle around a specified pivot

        Parameters
        ----------
        r0 : float
            The R location of the pivot
        z0 : float
            The Z location of the pivot
        angle : float
            The angle of the rotation in degrees as measured from the z axis
        """
        angle = pi / 180.0 * angle
        cost = cos(angle)
        sint = sin(angle)
        for c_obj in self._obj_list:
            r, z = c_obj.loc
            newr = cost * (r - r0) + sint * (z - z0) + r0
            newz = -sint * (r - r0) + cost * (z - z0) + z0
            c_obj.loc = (newr, newz)
        self.update_patch()

    def build_patchargs(self, **kwargs):
        """Build argument tuple for patchcls"""
        raise NotImplementedError("This method should be overridden in the child class")

    def rebuild(self, key, value):
        """Reset the CurrentGroup based on the key, value pairs passed in"""
        cls_dict = self.to_dict()
        cls_dict.pop('cls', None)
        if key not in cls_dict.keys():
            raise KeyError(f'{key} not in dict representing {type(self)}')
        cls_dict[key] = value
        self.reset(**cls_dict)

    def update_patch(self):
        """Update the patch for the CurrentGroup"""
        try:
            patchargs = self.build_patchargs(**self.patchargs_dict)
            self._patch = self.patchcls(*patchargs, **self.patchkwargs)
        except NotImplementedError:
            self._patch = None

    def plot_currents(self, ax, *args, **kwargs):
        """Plot the current locations for the CurrentGroup

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axes object for plotting the current locations
        *args : tuple
            Positional arguments to pass to Current.plot method
        **kwargs : dict, optional
            Keyword arguments to pass to Current.plot method
        """
        for c_obj in self._obj_list:
            c_obj.plot(ax, *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        """Plot the current locations for the CurrentGroup

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axes object for plotting the current locations
        *args : tuple
            Positional arguments to pass to Current.plot method
        **kwargs : dict, optional
            Keyword arguments to pass to Current.plot method
        """
        if kwargs.pop("plot_center", True):
            try:
                ax.plot(self.loc[0], self.loc[1], "co")
            except AttributeError:
                pass
        if kwargs.pop("plot_patch", True):
            ax.add_collection(PatchCollection([self.patch], match_original=True))
        if kwargs.pop("plot_currents", True):
            self.plot_currents(ax)

    def to_dict(self):
        """Represent the CurrentGroup as a dictionary"""
        cls_dict = {key.strip("_"): value for key, value in self.__dict__.items()}
        cls_dict.pop("obj_list")
        cls_dict.pop("patch")
        cls = str(self.__class__).split("'")[1]
        cls_dict.update({"rz_pts": self.rzdir[:, 0:2], "cls": cls})
        cls_dict.update(cls_dict.pop("patchkwargs"))
        return cls_dict

    @classmethod
    def from_dict(cls, cls_dict):
        """Create Current instance from a dictionary

        Parameters
        ----------
        cls_dict : dict
            The dictionary from which to construct a Current.
        """
        cls_str = cls_dict.pop("cls", None)
        return cls(**cls_dict)


class Magnet(CurrentGroup):
    """Represent a Rectangular cross-section dipole magnet with axisymmetric 
    surface currents.

    Parameters
    ----------
    rz_pts : iterable, optional
        Nx2 iterable representing R,Z current centroids. Defaults to None
    current : float, optional
        The current in all the current ring in amps, defaults to 1 amp.
    kwargs : matplotlib patch keyword arguments

    Attributes
    ----------
    loc : tuple
        The (R, Z) location of the centroid of the magnet.
    width : float, optional
        The width of the magnet. Defaults to 0.01 m.
    height : float, optional
        The height of the magnet. Defaults to 0.01 m.
    mu_hat : float, optional
        The angle of the magnetic moment of the magnet in degrees from the z
        axis. Defaults to 0 degrees clockwise from Z axis (i.e. north pole
        points in the +z direction).
    current_prof : integer or array_like
        The current profile along the side of the magnet. Defaults to
        np.ones(8) i.e. 8 equal surface currents per side.
    current : float
        The current in the magnet in amps.
    obj_list : list
        The list of Current objects that comprise the Magnet
    rzdir : np.array
        An Nx3 array whos rows are rzdir[i, :] = rloc, zloc, current which
        describe the current location and current value for each current in the
        Magnet
    patch : matplotlib.patches.Patch object
        The patch object representing the Magnet for plotting
    patchkwargs : dict
        The keyword arguments used for the patch attribute
    """

    def __init__(self, **kwargs):
        Magnet.reset(self, **kwargs)

    def reset(self, **kwargs):
        # set Magnet specific attributes before calling super constructor
        r0, z0 = kwargs.pop("loc", (1.0, 1.0))
        if r0 < 0:
            raise ValueError("Centroid of magnet, r0, must be >= 0")
        r0 = float(r0)
        z0 = float(z0)
        width = float(kwargs.pop("width", .01))
        height = float(kwargs.pop("height", .01))
        if not (width > 0 and height > 0):
            raise ValueError("width and height must be greater than 0")
        self._width = width
        self._height = height
        ## need to pop this now but save it for later
        mu_hat = kwargs.pop("mu_hat", 0)
        self._mu_hat = 0
        current_prof = kwargs.pop("current_prof", 10)
        if isinstance(current_prof, Number):
            current_prof = ones(current_prof)
        else:
            current_prof = array(current_prof)
        if not current_prof.size > 0:
            raise ValueError("current_prof array must have size > 0")
        self._current_prof = current_prof
        self._loc = (r0, z0)
        # start building super class relevant inputs
        # super_kwargs include rz_pts,current,patchcls,patchargs_dict, any matplotlib.patches kwarg
        current = kwargs.pop("current", 1)
        if not current > 0:
            raise ValueError("current must be > 0")
        self._current = current
        n = len(self._current_prof)
        dummy = ones(n)
        rpts = self._width / 2.0 * hstack((-1 * dummy, dummy))
        if n == 1:
            zpts = zeros(2)
        else:
            ztmp = linspace(-self._height / 2.0, self._height / 2.0, n)
            zpts = hstack((ztmp, ztmp))
        rz_pts = vstack((rpts + r0, zpts + z0)).T
        patchkwargs = {"closed": True, "fc": "w", "ec": "k", "zorder": 3}
        # All leftover kwargs get put into patchkwargs
        patchkwargs.update(kwargs)
        # Build kwargs for super constructor
        super_kwargs = {"rz_pts": rz_pts, "current": 1.0, "patchcls": Polygon, "patchargs_dict": {}}
        super_kwargs.update(patchkwargs)
        # builds CurrentGroup at loc with current = 1 for all current objs
        super(Magnet, self).__init__(**super_kwargs)
        # make left side currents negative (current setter overridden below)
        self.current = self._current
        # rotate according to muhat direction
        self.mu_hat = mu_hat

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, r0, z0):
        self.rebuild("loc", (r0, z0))

    @CurrentGroup.current.setter
    def current(self, new_current):
        # makes first half of obj_list have negative currents
        if new_current < 0:
            raise ValueError("current for Magnet class must be > 0")
        self._current = new_current
        n = len(self._obj_list) / 2
        for i, c_obj in enumerate(self._obj_list):
            c_obj.current = new_current * (-1) ** (i // n + 1)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, new_width):
        self.rebuild("width", new_width)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, new_height):
        self.rebuild("height", new_height)

    @property
    def current_prof(self):
        return self._current_prof

    @current_prof.setter
    def current_prof(self, new_prof):
        self.rebuild("current_prof", new_prof)

    @property
    def mu_hat(self):
        return self._mu_hat

    @mu_hat.setter
    def mu_hat(self, mu_hat):
        self.rotate(mu_hat - self._mu_hat)

    @property
    def rzdir(self):
        return array([c_obj.loc + (sign(c_obj._current)) for c_obj in self._obj_list], dtype="float32")

    def rotate(self, angle):
        """Rotate the magnet by a given angle around the magnet centroid

        Parameters
        ----------
        angle : float
            The angle of the rotation in degrees as measured from the z axis
        """
        r0, z0 = self._loc
        self._mu_hat += angle
        super(Magnet, self).rotate(r0, z0, angle)

    def translate(self, dr, dz):
        """Translate the magnet by the vector (dr, dz)

        Parameters
        ----------
        dr : float
            The displacement in the R direction for the translation
        dz : float
            The displacement in the Z direction for the translation
        """
        r0, z0 = self._loc
        self.loc = (r0 + dr, z0 + dz)

    def build_patchargs(self, **kwargs):
        """Build argument tuple for patchcls"""
        w = self._width / 2.0
        h = self._height / 2.0
        return (array([[-w, -h], [-w, h], [w, h], [w, -h]]),)

    def update_patch(self):
        """Update the patch for the magnet"""
        super(Magnet, self).update_patch()
        r0, z0 = self._loc
        self._patch.set_transform(Affine2D().translate(r0, z0).rotate_deg_around(r0, z0, -self._mu_hat))

    def to_dict(self):
        "Represent the Magnet with a dictionary"""
        cls_dict = {key.strip("_"): value for key, value in self.__dict__.items()}
        cls_dict.pop('obj_list')
        cls_dict.pop('patch')
        cls_dict["cls"] = str(self.__class__).split("'")[1]
        cls_dict.update(cls_dict.pop("patchkwargs"))
        return cls_dict


class CurrentArray(CurrentGroup):
    """A rectangular current array

    Parameters
    ----------
    loc : tuple, optional
        The (R, Z) location of the centroid of the CurrentArray. Defaults to
        (1.0m, 1.0m).
    current : float, optional
        The current in each current of the CurrentArray in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float, optional
        The number of current filaments in the R direction. Defaults to 10.
    nz : float, optional
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float, optional
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float, optional
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    angle : float, optional
        The angle of the CurrentArray if it is not aligned with the RZ
        coordinate system. The angle is measured in degrees from the z
        axis. Defaults to 0 degrees.
    patchcls : matplotlib.patches.Patch type
        The patch object class representing the CurrentArray for plotting

    Attributes
    ----------
    loc : tuple
        The (R, Z) location of the centroid of the CurrentArray. Defaults to
        (1.0m, 1.0m).
    current : float
        The current in each current of the CurrentArray in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    angle : float
        The angle of the CurrentArray if it is not aligned with the RZ
        coordinate system. The angle is measured in degrees from the z
        axis. Defaults to 0 degrees.
    obj_list : list
        The list of Current objects that comprise the CurrentArray
    rzdir : np.array
        An Nx3 array whos rows are rzdir[i, :] = rloc, zloc, current which
        describe the current location and current value for each current in the
        CurrentArray
    patchcls : matplotlib.patches.Patch type
        The patch object class representing the CurrentArray for plotting
    patch : matplotlib.patches.Patch object
        The patch object representing the CurrentArray for plotting
    patchkwargs : dict
        The keyword arguments used for the patch attribute
    """

    def __init__(self, **kwargs):
        CurrentArray.reset(self, **kwargs)

    def reset(self, **kwargs):
        r0, z0 = kwargs.pop("loc", (1.0, 0.0))
        current = kwargs.pop("current", 1.0)
        nr = kwargs.pop("nr", 10)
        nz = kwargs.pop("nz", 10)
        dr = kwargs.pop("dr", .01)
        dz = kwargs.pop("dz", .01)
        angle = float(kwargs.pop("angle", 0))
        patchcls = kwargs.pop("patchcls", Polygon)
        self._loc = (r0, z0)
        self._nr = nr
        self._nz = nz
        self._dr = dr
        self._dz = dz
        self._angle = 0
        rstart, rend = r0 - (nr - 1) * dr / 2.0, r0 + (nr - 1) * dr / 2.0
        zstart, zend = z0 - (nz - 1) * dz / 2.0, z0 + (nz - 1) * dz / 2.0
        rpts, zpts = linspace(rstart, rend, nr), linspace(zstart, zend, nz)
        rrpts, zzpts = meshgrid(rpts, zpts)
        rz_pts = vstack((rrpts.flatten(), zzpts.flatten())).T
        super_kwargs = {"rz_pts": rz_pts, "current": current, "patchcls": patchcls}
        super_kwargs.update(kwargs)
        super(CurrentArray, self).__init__(**super_kwargs)
        self.angle = angle

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc):
        r0, z0 = loc
        self.rebuild("loc", (r0, z0))

    @property
    def nr(self):
        return self._nr

    @nr.setter
    def nr(self, new_nr):
        self.rebuild("nr", new_nr)

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, new_nz):
        self.rebuild("nz", new_nz)

    @property
    def dr(self):
        return self._dr

    @dr.setter
    def dr(self, new_dr):
        self.rebuild("dr", new_dr)

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, new_dz):
        self.rebuild("dz", new_dz)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, new_angle):
        deg = new_angle - self._angle
        self._angle = new_angle
        r0, z0 = self._loc
        super(CurrentArray, self).rotate(r0, z0, deg)

    def translate(self, dr, dz):
        """Translate the CurrentArray by the vector (dr, dz)

        Parameters
        ----------
        dr : float
            The displacement in the R direction for the translation
        dz : float
            The displacement in the Z direction for the translation
        """
        r0, z0 = self._loc
        self.loc = (r0 + dr, z0 + dz)

    def rotate(self, angle):
        """Rotate the CurrentArray by a given angle around the centroid

        Parameters
        ----------
        angle : float
            The angle of the rotation in degrees as measured from the z axis
        """
        r0, z0 = self._loc
        super(CurrentArray, self).rotate(r0, z0, angle)

    def build_patchargs(self, **kwargs):
        """Build argument tuple for patchcls"""
        r0, z0 = self._loc
        w = (self._nr - 1) * self._dr / 2.0
        h = (self._nz - 1) * self._dz / 2.0
        return (array([[-w, -h], [-w, h], [w, h], [w, -h]]),)

    def update_patch(self):
        """Update the patch for the CurrentArray"""
        super(CurrentArray, self).update_patch()
        r0, z0 = self._loc
        angle = self._angle
        trnsf = Affine2D().translate(r0, z0).rotate_deg_around(r0, z0, -angle)
        self._patch.set_transform(trnsf)

    def to_dict(self):
        """Represent the CurrentArray with a dictionary"""
        cls_dict = {key.strip("_"): value for key, value in self.__dict__.items()}
        cls_dict.pop("obj_list")
        cls_dict.pop("patch")
        cls_dict["cls"] = str(self.__class__).split("'")[1]
        cls_dict.update(cls_dict.pop("patchkwargs"))
        return cls_dict


class MagnetGroup(object):
    """Represent a group of dipole magnets.

    Parameters
    ----------
    rz_pts : iterable, optional
        Nx2 iterable representing R,Z current centroids. Defaults to None
    mu_hats : iterable of float, optional
        A list of the angles of the magnetic moment for each magnet in
        degrees from the z axis. Defaults to None.
    current : float, optional
        The current in all the current ring in amps, defaults to 1 amp.
    kwargs : dict, optional
        A dictionary holding the keyword arguments for each pleiades.Magnet
        object.

    Attributes
    ----------
    current : float
        The current in the magnet in amps.
    obj_list : list
        The list of Current objects that comprise the Magnet
    rzdir : np.array
        An Nx3 array whos rows are rzdir[i, :] = rloc, zloc, current which
        describe the current location and current value for each current in the
        Magnet
    patches : list of matplotlib.patches.Patch objects
        The patch objects representing the MagnetGroup for plotting
    """
    def __init__(self, **kwargs):
        MagnetGroup.reset(self, **kwargs)

    def reset(self, **kwargs):
        rz_pts = array(kwargs.pop("rz_pts", [(1, 1)]))
        mu_hats = array(kwargs.pop("mu_hats", None))
        if mu_hats is None:
            mu_hats = zeros(len(rz_pts))
        current = float(kwargs.get("current", 1))
        self._current = current
        n, d = rz_pts.shape
        if not (d == 2 and n >= 1):
            raise ValueError(f'rz_pts shape: {(n, d)} is invalid, must be Nx2')
        self.obj_list = [Magnet(loc=(r, z), mu_hat=mhat, **kwargs)
                         for r, z, mhat in vstack((rz_pts.T, mu_hats)).T]

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, new_current):
        if new_current < 0:
            raise ValueError("current for MagnetGroup class must be > 0")
        self._current = new_current
        for i, m_obj in enumerate(self._obj_list):
            m_obj.current = new_current

    @property
    def obj_list(self):
        return self._obj_list

    @obj_list.setter
    def obj_list(self, new_obj_list):
        assert all([type(m_obj) == Magnet for m_obj in new_obj_list]), "All objects must be of type fields.core.Magnet"
        self._obj_list = new_obj_list

    @property
    def rzdir(self):
        return array([c_obj.loc + (sign(c_obj._current),) for m_obj in self._obj_list for c_obj in m_obj._obj_list],
                     dtype="float32")

    @property
    def patches(self):
        return [m_obj.patch for m_obj in self._obj_list]

    def update_patch(self):
        """Udate the patches for the MagnetGroup"""
        for m_obj in self._obj_list:
            m_obj.update_patch()

    def plot(self, ax):
        """Plot magnets

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axis on which to plot the magnets
        """
        for m_obj in self._obj_list:
            m_obj.plot(ax)

    def plot_currents(self, ax):
        """Plot current locations with markers for +/-

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axis on which to plot the current location
        """
        for m_obj in self._obj_list:
            m_obj.plot_currents(ax)

    def rebuild(self, key, value):
        """Reset the MAgnetGroup based on the key, value pair passed in"""
        for m_obj in self._obj_list:
            m_obj.rebuild(key, value)

    def to_dict(self):
        """Represent the MagnetGroup as a dictionary"""
        magnets = {i: m_obj.to_dict() for i, m_obj in enumerate(self._obj_list)}
        cls_dict = {"magnets": magnets}
        cls_dict["current"] = self._current
        cls_dict["cls"] = str(self.__class__).split("'")[1]
        return cls_dict

    @classmethod
    def from_dict(cls, cls_dict):
        """Create MagnetGroup instance from a dictionary

        Parameters
        ----------
        cls_dict : dict
            The dictionary from which to construct a Current.
        """
        inst = cls()
        current = cls_dict.pop("current")
        inst.obj_list = [Magnet.from_dict(cls_dict) for key, cls_dict in cls_dict.pop("magnets")]
        inst.current = current
        return inst


class Component(object):
    """A Container for representing multiple sets of objects and assigning a
    Green's function to the object. Components are like HH coils or Mirror
    Coils or Vessel Magnets. This is the minimum scale that has its own Green's
    function - one for each group.

    Attributes
    ----------
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    num_groups : int
        Number of groups that make up the Component
    currents : list of float
        A list of currents representing the current in each group in the
        Component
    nprocs : int
        The number of processors to use to compute the Green's functions
    patches : list of matplotlib.patches.Patch objects
        A list of all the patches that represent the Component
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields
    gpsi : np.array
        Green's function for magnetic flux psi
    gBR : np.array
        Green's function for magnetic field component BR
    gBZ : np.array
        Green's function for magnetic field component BZ
    psi : np.array
        Magnetic flux evaluated on the grid
    BR : np.array
        Magnetic field component BR evaluated on the grid
    BZ : np.array
        Magnetic field component BZ evaluated on the grid

    """
    def __init__(self):
        self._groups = None
        self._labels = None
        self._num_groups = 0
        self._currents = None
        self._nprocs = None
        self._patch_mask = None
        self.grid = None

    def compute_greens(self):
        """Compute the Green's functions for flux (psi) and BR and BZ"""
        simplefilter("ignore", RuntimeWarning)
        proc_max = cpu_count()
        m, n = self.grid.size, self._num_groups
        gpsi = zeros((m, n))
        gBR = zeros((m, n))
        gBZ = zeros((m, n))
        R = self.grid.R1D
        Z = self.grid.Z1D
        for i, (group, nprocs) in enumerate(zip(self._groups, self._nprocs)):
            rzdir = group.rzdir
            procs = []
            pid_list = []
            out_q = Queue()
            if nprocs > proc_max:
                nprocs = proc_max
            chunksize = int(ceil(rzdir.shape[0] / float(nprocs)))
            for j in range(nprocs):
                p = Process(target=_get_greens, args=(R, Z, rzdir[j * chunksize:(j + 1) * chunksize, :]),
                            kwargs={"out_q": out_q})
                procs.append(p)
                p.start()
                pid_list.append(str(p.pid))

            for k in range(nprocs):
                g_tup = out_q.get()
                gpsi[:, i] += g_tup[0]
                gBR[:, i] += g_tup[1]
                gBZ[:, i] += g_tup[2]

            for p in procs:
                p.join()

        self._gpsi = gpsi
        self._gBR = gBR
        self._gBZ = gBZ

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, new_groups):
        self._groups = new_groups
        self._num_groups = len(self._groups)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        # del old attributes
        try:
            for label in self._labels:
                delattr(self, label)
        except TypeError:
            pass
        # then make new ones
        self._labels = new_labels
        for label, group in zip(self._labels, self._groups):
            setattr(self, label, group)

    @property
    def num_groups(self):
        return self._num_groups

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, newgrid):
        if newgrid is None:
            self._grid = None
            self._gpsi = None
            self._gBR = None
            self._gBZ = None
        else:
            self._grid = newgrid
            self.compute_greens()

    @property
    def currents(self):
        return array([group.current for group in self._groups])

    @currents.setter
    def currents(self, new_currents):
        new_currents = array(new_currents, dtype="float32")
        assert len(new_currents) == len(self._groups), "length of groups and currents must match"
        for group, cur in zip(self._groups, new_currents):
            group.current = cur

    @property
    def nprocs(self):
        return self._nprocs

    @nprocs.setter
    def nprocs(self, new_nprocs):
        if len(new_nprocs) != self._num_groups:
            raise ValueError("length of nprocs must match current number of groups")
        self._nprocs = new_nprocs

    @property
    def patch_mask(self):
        return self._patch_mask

    @patch_mask.setter
    def patch_mask(self, new_mask):
        if len(new_mask) != self._num_groups:
            raise ValueError("length of patch_mask must match current number of groups")
        self._patch_mask = new_mask

    @property
    def patches(self):
        return [group.patch for group, mask in zip(self._groups, self._patch_mask) if not mask]

    @property
    def gpsi(self):
        return self._gpsi

    @property
    def gBR(self):
        return self._gBR

    @property
    def gBZ(self):
        return self._gBZ

    @property
    def psi(self):
        return (self._gpsi.dot(self.currents)).reshape(self._grid.shape)

    @property
    def BR(self):
        return (self._gBR.dot(self.currents)).reshape(self._grid.shape)

    @property
    def BZ(self):
        return (self._gBZ.dot(self.currents)).reshape(self._grid.shape)

    def plot_currents(self, ax, *args, **kwargs):
        """Plot current locations with markers for +/- for all the objects in
        the Component

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axis on which to plot the current location
        """
        for group in self.groups:
            group.plot_currents(ax, *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        """Plot each group including patches for all the objects in the
        Component

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axis on which to plot the current location
        """
        for group in self.groups:
            group.plot(ax, *args, **kwargs)

    def update_patches(self):
        """Update the patches for the Component"""
        for group in self._groups:
            group.update_patch()

    def update(self):
        """Update the patches and Green's function for the component"""
        self.compute_greens()
        self.update_patches()

    def to_dict(self):
        """Represent the component as a dictionary"""
        cls_dict = dict(key.strip("_"): value
                        for key, value in self.__dict__.items())
        cls_dict.pop(label, None)
        for group, label in zip(self._groups, self._labels):
            cls_dict[label] = group.to_dict()
        cls_dict["cls"] = str(self.__class__).split("'")[1]
        return cls_dict

    @classmethod
    def from_dict(cls, cls_dict):
        """Create Component from a dictionary

        Parameters
        ----------
        cls_dict : dict
            The dictionary from which to construct a Component.
        """
        labels = cls_dict.get("labels")
        comp_cls = get_class(cls_dict.pop("cls"))
        groups = []
        for label in labels:
            group_dict = cls_dict.pop(label)
            sub_cls = get_class(group_dict.pop("cls"))
            groups.append(sub_cls.from_dict(**group_dict))
        gpsi = cls_dict.pop("gpsi")
        gBR = cls_dict.pop("gBR")
        gBZ = cls_dict.pop("gBZ")
        grid_dict = cls_dict.pop("grid")
        obj = comp_cls(**cls_dict)
        obj._gpsi = gpsi
        obj._gBR = gBR
        obj._gBZ = gBZ
        obj._grid = None
        return obj


class Coil(Component):
    """A component representing a single Coil

    Parameters
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    current : float
        The current in each current of the Coil in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields

    Attributes
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    current : float
        The current in each current of the Coil in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    num_groups : int
        Number of groups that make up the Component
    currents : list of float
        A list of currents representing the current in each group in the
        Component
    nprocs : int
        The number of processors to use to compute the Green's functions
    patches : list of matplotlib.patches.Patch objects
        A list of all the patches that represent the Component
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields
    gpsi : np.array
        Green's function for magnetic flux psi
    gBR : np.array
        Green's function for magnetic field component BR
    gBZ : np.array
        Green's function for magnetic field component BZ
    psi : np.array
        Magnetic flux evaluated on the grid
    BR : np.array
        Magnetic field component BR evaluated on the grid
    BZ : np.array
        Magnetic field component BZ evaluated on the grid
    """
    def __init__(self, **kwargs):
        super(Coil, self).__init__()
        r0 = float(kwargs.pop("r0", 1))
        z0 = float(kwargs.pop("z0", 1))
        nr = kwargs.pop("nr", 10)
        nz = kwargs.pop("nz", 10)
        dr = kwargs.pop("dr", .01)
        dz = kwargs.pop("dz", .01)
        labels = kwargs.pop("labels", None)
        currents = array(kwargs.pop("currents", [1]), dtype="float")
        nprocs = kwargs.pop("nprocs", [4])
        patch_mask = kwargs.pop("patch_mask", [0])
        grid = kwargs.pop("grid", None)
        self._r0 = r0
        self._z0 = z0
        self._nr = nr
        self._nz = nz
        self._dr = dr
        self._dz = dz
        coil1 = CurrentArray(loc=(r0, z0), nr=nr, nz=nz, dz=dz, dr=dr, **kwargs)
        self.groups = [coil1]
        if labels is None:
            labels = ["group{0}".format(i) for i in range(len(self.groups))]
        self.labels = labels
        self.currents = currents
        self.nprocs = nprocs
        self.patch_mask = patch_mask
        self.grid = grid

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, new_z0):
        r0 = self._r0
        self._z0 = new_z0
        self.groups[0].loc = (r0, new_z0)

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, new_r0):
        z0 = self._z0
        self._r0 = new_r0
        self.groups[0].loc = (new_r0, z0)

    @property
    def nr(self):
        return self._nr

    @nr.setter
    def nr(self, new_nr):
        self._nr = new_nr
        for c_arr in self._groups:
            c_arr.nr = new_nr

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, new_nz):
        self._nz = new_nz
        for c_arr in self._groups:
            c_arr.nz = new_nz

    @property
    def dr(self):
        return self._dr

    @dr.setter
    def dr(self, new_dr):
        self._dr = new_dr
        for c_arr in self._groups:
            c_arr.dr = new_dr

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, new_dz):
        self._dz = new_dz
        for c_arr in self._groups:
            c_arr.dz = new_dz


class CoilPack(Component):
    """A component representing a single CoilPack

    Parameters
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    current : float
        The current in each current of the Coil in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields

    Attributes
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    current : float
        The current in each current of the Coil in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    num_groups : int
        Number of groups that make up the Component
    currents : list of float
        A list of currents representing the current in each group in the
        Component
    nprocs : int
        The number of processors to use to compute the Green's functions
    patches : list of matplotlib.patches.Patch objects
        A list of all the patches that represent the Component
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields
    gpsi : np.array
        Green's function for magnetic flux psi
    gBR : np.array
        Green's function for magnetic field component BR
    gBZ : np.array
        Green's function for magnetic field component BZ
    psi : np.array
        Magnetic flux evaluated on the grid
    BR : np.array
        Magnetic field component BR evaluated on the grid
    BZ : np.array
        Magnetic field component BZ evaluated on the grid
    """
    def __init__(self, **kwargs):
        super(CoilPack, self).__init__()
        r0 = float(kwargs.pop("r0", 1))
        z0 = float(kwargs.pop("z0", 1))
        nr = kwargs.pop("nr", 10)
        nz = kwargs.pop("nz", 10)
        dr = kwargs.pop("dr", .01)
        dz = kwargs.pop("dz", .01)
        labels = kwargs.pop("labels", None)
        currents = array(kwargs.pop("currents", (1,)), dtype="float")
        nprocs = kwargs.pop("nprocs", [4])
        patch_mask = kwargs.pop("patch_mask", [0])
        grid = kwargs.pop("grid", None)
        self._r0 = r0
        self._z0 = z0
        self._nr = nr
        self._nz = nz
        self._dr = dr
        self._dz = dz
        coil = CurrentArray(loc=(r0, z0), nr=nr, nz=nz, dz=dz, dr=dr, **kwargs)
        self.groups = [coil]
        if labels is None:
            labels = ["group{0}".format(i) for i in range(len(self.groups))]
        self.labels = labels
        self.currents = currents
        self.nprocs = nprocs
        self.patch_mask = patch_mask
        self.grid = grid

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, new_z0):
        r0 = self._r0
        self._z0 = new_z0
        self.groups[0].loc = (r0, new_z0)
        self.update()

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, new_r0):
        z0 = self._z0
        self._r0 = new_r0
        self.groups[0].loc = (new_r0, z0)
        self.update()

    @property
    def nr(self):
        return self._nr

    @nr.setter
    def nr(self, new_nr):
        self._nr = new_nr
        for c_arr in self._groups:
            c_arr.nr = new_nr
        self.update()

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, new_nz):
        self._nz = new_nz
        for c_arr in self._groups:
            c_arr.nz = new_nz
        self.update()

    @property
    def dr(self):
        return self._dr

    @dr.setter
    def dr(self, new_dr):
        self._dr = new_dr
        for c_arr in self._groups:
            c_arr.dr = new_dr
        self.update()

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, new_dz):
        self._dz = new_dz
        for c_arr in self._groups:
            c_arr.dz = new_dz
        self.update()


class ZSymmCoilSet(Component):
    """A component representing a coil set that is symmetric about Z=0

    Parameters
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    current : float
        The current in each current of the Coil in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields

    Attributes
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    current : float
        The current in each current of the Coil in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    num_groups : int
        Number of groups that make up the Component
    currents : list of float
        A list of currents representing the current in each group in the
        Component
    nprocs : int
        The number of processors to use to compute the Green's functions
    patches : list of matplotlib.patches.Patch objects
        A list of all the patches that represent the Component
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields
    gpsi : np.array
        Green's function for magnetic flux psi
    gBR : np.array
        Green's function for magnetic field component BR
    gBZ : np.array
        Green's function for magnetic field component BZ
    psi : np.array
        Magnetic flux evaluated on the grid
    BR : np.array
        Magnetic field component BR evaluated on the grid
    BZ : np.array
        Magnetic field component BZ evaluated on the grid
    """
    def __init__(self, **kwargs):
        super(ZSymmCoilSet, self).__init__()
        r0 = float(kwargs.pop("r0", 1))
        z0 = float(kwargs.pop("z0", 1))
        nr = kwargs.pop("nr", 10)
        nz = kwargs.pop("nz", 10)
        dr = kwargs.pop("dr", .01)
        dz = kwargs.pop("dz", .01)
        labels = kwargs.pop("labels", None)
        currents = array(kwargs.pop("currents", (1, 1)), dtype="float")
        nprocs = kwargs.pop("nprocs", [4, 4])
        patch_mask = kwargs.pop("patch_mask", [0, 0])
        grid = kwargs.pop("grid", None)
        self._r0 = r0
        self._z0 = z0
        self._nr = nr
        self._nz = nz
        self._dr = dr
        self._dz = dz
        coil1 = CurrentArray(loc=(r0, -z0), nr=nr, nz=nz, dz=dz, dr=dr, **kwargs)
        coil2 = CurrentArray(loc=(r0, z0), nr=nr, nz=nz, dz=dz, dr=dr, **kwargs)
        self.groups = [coil1, coil2]
        if labels is None:
            labels = ["group{0}".format(i) for i in range(len(self.groups))]
        self.labels = labels
        self.currents = currents
        self.nprocs = nprocs
        self.patch_mask = patch_mask
        self.grid = grid

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, new_z0):
        r0 = self._r0
        self._z0 = new_z0
        self.groups[0].loc = (r0, -new_z0)
        self.groups[1].loc = (r0, new_z0)

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, new_r0):
        z0 = self._z0
        self._r0 = new_r0
        self.groups[0].loc = (new_r0, -z0)
        self.groups[1].loc = (new_r0, z0)

    @property
    def nr(self):
        return self._nr

    @nr.setter
    def nr(self, new_nr):
        self._nr = new_nr
        for c_arr in self._groups:
            c_arr.nr = new_nr

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, new_nz):
        self._nz = new_nz
        for c_arr in self._groups:
            c_arr.nz = new_nz

    @property
    def dr(self):
        return self._dr

    @dr.setter
    def dr(self, new_dr):
        self._dr = new_dr
        for c_arr in self._groups:
            c_arr.dr = new_dr

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, new_dz):
        self._dz = new_dz
        for c_arr in self._groups:
            c_arr.dz = new_dz


class HelmholtzCoil(ZSymmCoilSet):
    """A Helmholtz coil set

    Parameters
    ----------
    r0 : float, optional
        The radius of the Helmholtz coil set. Defaults to 1.0m
    z0 : float, optional
        The separation of the Helmholtz coil set. Defaults to 1.0m
    **kwargs : dict, optional
        Keyword arguments for pleiades.ZSymmCoilSet

    Attributes
    ----------
    r0 : float, optional
        The radius of the Helmholtz coil set. Defaults to 1.0m
    z0 : float, optional
        The separation of the Helmholtz coil set. Defaults to 1.0m
    current : float
        The current in each current of the Coil in amps (i.e power
        supply current). Defaults to 1 amp.
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    groups : list
        A list of all the groups that comprise this Component
    labels : list of str
        A list of all the names for the groups that comprise this Component.
        Each label is also accessible as an attribute on the Component as well.
    num_groups : int
        Number of groups that make up the Component
    currents : list of float
        A list of currents representing the current in each group in the
        Component
    nprocs : int
        The number of processors to use to compute the Green's functions
    patches : list of matplotlib.patches.Patch objects
        A list of all the patches that represent the Component
    patch_mask : iterable of bool
        A list of booleans of the same length as the number of groups where True
        indicates to hide the patch for that particular group.
    grid : pleiades.Grid instance
        A grid on which to compute the Green's functions for flux and magnetic
        fields
    gpsi : np.array
        Green's function for magnetic flux psi
    gBR : np.array
        Green's function for magnetic field component BR
    gBZ : np.array
        Green's function for magnetic field component BZ
    psi : np.array
        Magnetic flux evaluated on the grid
    BR : np.array
        Magnetic field component BR evaluated on the grid
    BZ : np.array
        Magnetic field component BZ evaluated on the grid
    """
    def __init__(self, **kwargs):
        r0 = float(kwargs.pop("r0", 1))
        # throw away z0 if specified
        z0 = float(kwargs.pop("z0", 1))
        super(HelmholtzCoil, self).__init__(r0=r0, z0=r0 / 2.0, **kwargs)

    @ZSymmCoilSet.z0.setter
    def z0(self, new_z0):
        self._z0 = new_z0
        self._r0 = new_z0 * 2
        self.groups[0].loc = (self._r0, -self._z0)
        self.groups[1].loc = (self._r0, self._z0)

    @ZSymmCoilSet.r0.setter
    def r0(self, new_r0):
        self._r0 = new_r0
        self._z0 = new_r0 / 2.0
        self.groups[0].loc = (self._r0, -self._z0)
        self.groups[1].loc = (self._r0, self._z0)


class Configuration(object):
    """A container for a full configuration of magnets for an experiment

    Parameters
    ----------
    components : list
        A list of pleiades.Component objects to be added to the configuration
    labels : list of str
        A list of the names for the components being added
    filename : str
        A filename for the Configuration
    grid : pleiades.Grid object
        A grid on which to compute Green's functions and fields
    artists :
        A list of matplotlib patch objects 

    Attributes
    ----------
    grid : pleiades.Grid object
        A grid on which to compute Green's functions and fields
    R : np.array
        The R locations of the grid
    Z : np.array
        The Z locations of the grid
    psi : np.array
        The psi values on the grid
    BR : np.array
        The BR values on the grid
    BZ : np.array
        The BZ values on the grid
    patches : list
        A list of patch objects for the configuration
    patch_coll : matplotlib.patches.PatchCollection
        A patch collection for easier adding to matplotlib axes
    """
    def __init__(self, **kwargs):
        self.components = kwargs.pop("components", [])
        self.labels = kwargs.pop("labels", [])
        self.filename = kwargs.pop("filename", None)
        for comp, label in zip(self.components, self.labels):
            setattr(self, label, comp)
        self.grid = kwargs.pop("grid", None)
        self.artists = kwargs.pop("artists", [])

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        for comp in self.components:
            comp.grid = grid

    @property
    def R(self):
        return self._grid.R

    @property
    def Z(self):
        return self._grid.Z

    @property
    def psi(self):
        return sum([comp.psi for comp in self.components], axis=0)

    @property
    def BR(self):
        return sum([comp.BR for comp in self.components], axis=0)

    @property
    def BZ(self):
        return sum([comp.BZ for comp in self.components], axis=0)

    @property
    def patches(self):
        plist = [c.patches for c in self.components]
        plist = [p for sublist in plist for p in sublist]
        plist.extend(self.artists)
        return plist

    @property
    def patch_coll(self):
        return PatchCollection(self.patches, match_original=True)

    def add_component(self, component, label):
        self.components.append(component)
        self.labels.append(label)
        setattr(self, label, component)

    def reset_grid(self):
        self.grid = self._grid

    def update(self):
        self.reset_grid()
        for comp in self.components:
            comp.update_patches()

    def plot_currents(self, ax):
        for comp in self.components:
            for group in comp.groups:
                group.plot(ax)

    def plot_psi(self, ax, *args, **kwargs):
        return ax.contour(self.grid.R, self.grid.Z, self.psi, *args, **kwargs)

    def plot_modB(self, ax, *args, **kwargs):
        return ax.contour(self.grid.R, self.grid.Z, sqrt(self.BR ** 2 + self.BZ ** 2), *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        for comp in self.components:
            comp.plot(ax, *args, **kwargs)
        ax.add_collection(PatchCollection(self.artists, match_original=True))

    def save(self, filename=None):
        self.update()
        if filename is None:
            if self.filename is None:
                raise ValueError("I can't find a filename to save to")
            else:
                save_config(self, self.filename)
        else:
            save_config(self, filename)


def load_config(filename):
    """Load a configuration from a pickle file"""
    if filename.lower().endswith(('.p', '.pickle')):
        with open(filename, "r") as f:
            config = pickle.load(f)
    elif filename.lower().endswith(('.h5', '.hdf5')):
        raise NotImplementedError("HDF5 compatibility not implemented yet")
    config.filename = filename
    return config


def save_config(config, filename):
    """Save a configuration to a pickle file"""
    config.update()
    if filename.lower().endswith(('.p', '.pickle')) or '.' not in filename:
        if '.' not in filename:
            filename += '.p'
        with open(filename, "w") as f:
            pickle.dump(config, f)

    elif filename.lower().endswith(('.h5', '.hdf5')):
        raise NotImplementedError("HDF5 compatibility not implemented yet")
    else:
        raise ValueError("Unsupported file extension")


def _get_greens(R, Z, rzdir, out_q=None):
    """Helper function for computing Green's functions

    Parameters
    ----------
    R : np.array
        A 1D np.array representing the R positions of the grid
    Z : np.array
        A 1D np.array representing the Z positions of the grid
    rzdir : np.array
        An Nx3 np.array representing the r, z positions and sign of the current
        in all the current filaments.
    out_q: multiprocessing.Queue object?
        Internally used for faster processing of Green's functions
    """
    simplefilter("ignore", RuntimeWarning)
    n = len(R)
    gpsi = zeros(n)
    gBR = zeros(n)
    gBZ = zeros(n)
    R2 = R ** 2
    mu_0 = 4 * pi * 10 ** -7
    pre_factor = mu_0 / (4 * pi)
    for r0, z0, csign in rzdir:
        if isclose(r0, 0, rtol=0, atol=1E-12):
            continue
        fac0 = (Z - z0) ** 2
        d = sqrt(fac0 + (R + r0) ** 2)
        d_ = sqrt(fac0 + (R - r0) ** 2)
        k_2 = 4 * R * r0 / d ** 2
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d_ ** 2 * d
        fac1 = d_ ** 2 * K
        fac2 = (fac0 + R2 + r0 ** 2) * E
        gpsi_tmp = csign * pre_factor * R * r0 / d * 4 / k_2 * ((2 - k_2) * K - 2 * E)
        gpsi_tmp[~isfinite(gpsi_tmp)] = 0
        gpsi += gpsi_tmp
        gBR_tmp = -2 * csign * pre_factor * (Z - z0) * (fac1 - fac2) / (R * denom)
        gBR_tmp[~isfinite(gBR_tmp)] = 0
        gBR += gBR_tmp
        gBZ_tmp = 2 * csign * pre_factor * (fac1 - (fac2 - 2 * r0 ** 2 * E)) / denom
        gBZ_tmp[~isfinite(gBZ_tmp)] = 0
        gBZ += gBZ_tmp
    out_tup = (gpsi, gBR, gBZ)
    if out_q is None:
        return out_tup
    out_q.put(out_tup)


def get_greens(R, Z, rzdir, out_q=None, out_idx=None):
    """Compute Green's functions for psi, BR, and BZ

    Parameters
    ----------
    R : np.array
        A 1D np.array representing the R positions of the grid
    Z : np.array
        A 1D np.array representing the Z positions of the grid
    rzdir : np.array
        An Nx3 np.array representing the r, z positions and sign of the current
        in all the current filaments.
    out_q: multiprocessing.Queue object?
        Internally used for faster processing of Green's functions
    out_idx: int?
        Internally used for faster processing of Green's functions

    Returns
    -------
    out_tup : tuple
        Tuple of 3 elements (gpsi, gBR, gBZ) for the 3 Green's functions
    """
    simplefilter("ignore", RuntimeWarning)
    m, n = len(R), len(rzdir)
    gpsi = zeros((m, n))
    gBR = zeros((m, n))
    gBZ = zeros((m, n))
    R2 = R ** 2
    mu_0 = 4 * pi * 10 ** -7
    pre_factor = mu_0 / (4 * pi)
    for i, (r0, z0, csign) in enumerate(rzdir):
        if isclose(r0, 0, rtol=0, atol=1E-12):
            continue
        fac0 = (Z - z0) ** 2
        d = sqrt(fac0 + (R + r0) ** 2)
        d_ = sqrt(fac0 + (R - r0) ** 2)
        k_2 = 4 * R * r0 / d ** 2
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d_ ** 2 * d
        fac1 = d_ ** 2 * K
        fac2 = (fac0 + R2 + r0 ** 2) * E
        gpsi_tmp = csign * pre_factor * R * r0 / d * 4 / k_2 * ((2 - k_2) * K - 2 * E)
        gpsi_tmp[~isfinite(gpsi_tmp)] = 0
        gpsi[:, i] = gpsi_tmp
        gBR_tmp = -2 * csign * pre_factor * (Z - z0) * (fac1 - fac2) / (R * denom)
        gBR_tmp[~isfinite(gBR_tmp)] = 0
        gBR[:, i] = gBR_tmp
        gBZ_tmp = 2 * csign * pre_factor * (fac1 - (fac2 - 2 * r0 ** 2 * E)) / denom
        gBZ_tmp[~isfinite(gBZ_tmp)] = 0
        gBZ[:, i] = gBZ_tmp
    out_tup = (gpsi, gBR, gBZ)
    if out_q is None:
        return out_tup
    else:
        if out_idx is None:
            raise ValueError("I don't know where to put this output, please specify out_idx")
        out_q.put((out_idx,) + out_tup)


def compute_greens(R, Z, rzdir=None, nprocs=1):
    """Compute Green's functions for psi, BR, and BZ

    Parameters
    ----------
    R : np.array
        A 1D np.array representing the R positions of the grid
    Z : np.array
        A 1D np.array representing the Z positions of the grid
    rzdir : np.array
        An Nx3 np.array representing the r, z positions and sign of the current
        in all the current filaments.
    out_q: multiprocessing.Queue object?
        Internally used for faster processing of Green's functions
    out_idx: int?
        Internally used for faster processing of Green's functions

    Returns
    -------
    out_tup : tuple
        Tuple of 3 elements (gpsi, gBR, gBZ) for the 3 Green's functions
    """
    simplefilter("ignore", RuntimeWarning)
    proc_max = cpu_count()
    if rzdir is None:
        rzdir = vstack((R, Z, ones(len(R)))).T
    m, n = len(R), len(rzdir)
    gpsi = zeros((m, n))
    gBR = zeros((m, n))
    gBZ = zeros((m, n))
    if nprocs > proc_max:
        nprocs = proc_max
    procs = []
    out_q = Queue()
    chunksize = int(ceil(rzdir.shape[0] / float(nprocs)))
    print(chunksize)
    for i in range(nprocs):
        p = Process(target=get_greens, args=(R, Z, rzdir[i * chunksize:(i + 1) * chunksize, :]),
                    kwargs={"out_q": out_q, "out_idx": i})
        procs.append(p)
        p.start()

    for j in range(nprocs):
        print("getting g_tup #: {0}".format(j))
        g_tup = out_q.get()
        idx = g_tup[0]
        gpsi[:, idx * chunksize:(idx + 1) * chunksize] = g_tup[1]
        gBR[:, idx * chunksize:(idx + 1) * chunksize] = g_tup[2]
        gBZ[:, idx * chunksize:(idx + 1) * chunksize] = g_tup[3]

    for p in procs:
        p.join()

    return (gpsi, gBR, gBZ)


def get_class(cls_str):
    parts = cls_str.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m
