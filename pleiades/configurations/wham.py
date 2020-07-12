from pleiades import Device, RectangularCoil


class WHAM(Device):
    """The Device object representing the Wisconsin HTS Axisymmetric Mirror.

    Attributes
    ----------
    hts1 : RectangularCoil object
        A coil for the positive Z HTS mirror coil
    hts2 : RectangularCoil object
        A coil for the negative Z HTS mirror coil
    cc1 : RectangularCoil object
        A coil for the positive central cell coil
    cc2 : RectangularCoil object
        A coil for the negative central cell coil
    """

    def __init__(self):
        # Global default patch settings
        super().__init__()

        # Set HTS mirror coil default parameters
        r0, z0 = 0.25, 0.942
        dr, dz = 0.01, 0.01
        nr, nz = 38, 11
        self.hts1 = RectangularCoil(r0, z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.hts2 = RectangularCoil(r0, -z0, nr=nr, nz=nz, dr=dr, dz=dz)

        # Set central coil default parameters
        r0, z0 = 1.005, 0.205
        dr, dz = 0.025, 0.024
        nr, nz = 2, 5
        self.cc1 = RectangularCoil(r0, z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.cc2 = RectangularCoil(r0, -z0, nr=nr, nz=nz, dr=dr, dz=dz)



