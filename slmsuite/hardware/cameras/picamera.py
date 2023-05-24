"""
Hardware control of the pi camera by accessing remotely to the Raspberry pi controlling the
server via jsonrpclib. 
"""

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography.analysis import return_interp_arr
import logging
from jsonrpclib import Server
import pickle
import base64
import time
import numpy as np

class picamera(Camera):
    """
    Wrap of the picamera
    
    To check to hardware specs of the camera, visit
    www.raspberrypi.com/documentation/accessories/camera.html

    Attributes
    ----------

    cam : object
        The server of the camera which provides some handle that are connected to the the hardware.
    """

    # Class variable (same for all instances of Template) pointing to a singleton SDK.
    sdk = None

    def __init__(
        self,
        ip="192.168.222.128",
        port="6009",
        verbose=True,
        **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        ip : str
            The ip address of the server, default set to 192.168.222.121 which is the address of the current server.
        port: str
            The port used for the communication, default set to 6009 which is the port currently used.
        verbose : bool
            Whether or not to print extra information.
        kwargs
            timeout: default 5
            transpose_frame: default False
            rotation: default 0
            flip_up_down: default False
            flip_lr: default False
            background: default set to 0
            return_green_interp
        """
        #Initialize the ip and the port
        self.ip = ip
        self.port = port
        self.verbose = verbose
        # Initialize the properties from kwargs
        self.timeout_s = kwargs.get("timeout", 5)
        self.exp_time = kwargs.get("exposure_time", 100)
        self.rot = kwargs.get("rotation", "0")
        self.flip_ud = kwargs.get("flip_up_down", False)
        self.flip_lr = kwargs.get("flip_lr", False)
        self.bg = kwargs.get("background", 0)
        self.return_green_interp = kwargs.get("return_green_interp", False)
        
        # Opening a the server and connect to picamera
        http_link = "http://" + self.ip + ":" + self.port
        if verbose:
            print("Connecting to server " + http_link + "... ")
        self.cam = Server(http_link)
        if verbose:
            print("Connected!")

        # Initializing the width and height to the full resolution, and gather the framerate of the pi camera.
        self.width, self.height = (3280, 2464)
        self.frame_rate = self.cam.get_framerate()
        self.bitdepth = 10 # TODO: check whether I can obtain this from the camera directly
        
        # Define the min and max exposure time of the picamera
        self.exp_time_min = 9
        self.exp_time_max = 1000000/self.frame_rate

        # Set the exposure boundary
        self.exposure_bounds_s = (self.exp_time_min, self.exp_time_max)

        # Finally, use the superclass constructor to initialize other required variables.
        super().__init__(
            self.width,                       # TODO: Fill in proper functions.
            self.height,
            bitdepth=self.bitdepth,
            dx_um=1.12,
            dy_um=1.12,
            rot=self.rot,
            flipud=self.flip_ud,
            fliplr=self.flip_lr,
            name="PiCamera V2",
        )

        # Set the exposure time to the default exposure time
        self.set_exposure(self.exp_time)
        
        # Set the analog and digital gain to unity
        self.cam.set_analog_gain(1)
        self.cam.set_digital_gain(1)

        # Set the WOI of the camera to full resolution
        self.set_woi(woi=(0, self.width, 0, self.height))

        # Declare color channel offsets
        # Update color channel indices
        # [ry, rx], [gy, gx], [Gy, Gx], [by, bx] = self.cam.get_bayer_offsets()
        # self.bayer_offsets = [ry, rx], [gy, gx], [Gy, Gx], [by, bx]

        
    def close(self):
        """See :meth:`.Camera.close`."""
        raise NotImplementedError()
        self.cam.close()                                # TODO: Fill in proper function.
        del self.cam

    @staticmethod
    def info(verbose=True):
        """
        Discovers all cameras detected by the SDK.
        Useful for a user to identify the correct serial numbers / etc.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of serial numbers or identifiers.
        """
        raise NotImplementedError()
        serial_list = Template.sdk.get_serial_list()    # TODO: Fill in proper function.
        return serial_list

    ### Property Configuration ###

    def get_exposure(self):
        """See :meth:`.Camera.get_exposure`.
        Returns the exposure time of the camera in micro-seconds. A value of zero
        indicates that the exposure time is set automatically. This is de-activated
        by setting the exposure mode attribute to "off".
        See: :meth:`.tiqi_camera.get_shutter_speed`
        """
        exposure_time = self.cam.get_shutter_speed()
        assert exposure_time >= 0, "The exposure time is set automatically."

        return float(exposure_time)

    def set_exposure(self, exposure_us):
       """See :meth:`.Camera.set_exposure`.
        The PiCamera.shutter_speed (or the PiCamera.exposure_speed) attributes
        are given in micro-seconds (us).
        CAUTION: Apparently the minimum increment for the exposure time is 19 us.
        So the minimum time is 9 -> 28 -> 47 and so on
        """
       
       self.cam.set_shutter_speed_manual(exposure_us)

    def set_woi(self, crop_width=10, set_auto=0, woi=[]):
        """See :meth:`.Camera.set_woi`.
        
        Parameters
        ----------
        crop_width: int
            half-diameter of the square bounding box drawn around the maximum pixel
        set_auto: bool (int cuz tiqi_plugin)
            if False, set woi manually
        woi: list
            woi bbox in format (width_min, width_max, height_min, height_max)
        """

        self.cam.set_woi(crop_width=crop_width, set_auto=set_auto, woi=woi)

        # Extract updated values
        # Width always refers to the camera width (0-3280), and height to the camera height (0-2464)
        # That is, the returned image from the Pi is *designed* to always have shape (2464, 3280)
        width_min, width_max, height_min, height_max = self._get_woi()

        if width_min < 0 or height_min < 0:
            logging.warning("The current window of interest exceeds the camera limits. Choose smaller crop width.")
        if height_max > 2464 or width_max > 3280:
            logging.warning("The current window of interest exceeds the camera limits. Choose smaller crop width.")

        # Change shape of camera object to reflect WOI - format (height, width)
        if self.rot:
            self.shape = (width_max - width_min, height_max - height_min)
        else:
            self.shape = (height_max - height_min, width_max - width_min)

        # Update WOI attribute
        if self.rot:
            self.woi = (height_min, height_max, width_min, width_max)
        else:
            self.woi = (width_min, width_max, height_min, height_max)
            
        # Update color channel indices
        [ry, rx], [gy, gx], [Gy, Gx], [by, bx] = self.cam.get_bayer_offsets()
        self.bayer_offsets = [ry, rx], [gy, gx], [Gy, Gx], [by, bx]

    def _get_woi(self):
        """Returns the current window of interest (WOI).
        Out:
        ----
        woi: tuple
            woi bbox in format (width_min, width_max, height_min, height_max)
        """
        return self.cam.get_woi()

    def get_image(self, timeout_s=0.0, background=None, return_green_interp=None):
        """See :meth:`.Camera.get_image`.

        Parameters:
        -----------
        timeout_s: float
            wait time before taking an image

        Returns:
        --------
        frame: np.ndarray
            numpy array (cropped, background subtracted, raw, transformed)
        """

        # Wait for frame to be fetched
        time.sleep(timeout_s)

        st = time.time()

        # # String workaround - Pavel 02/20/2023
        frame_string = self.cam.get_frame_string(return_offsets=False, export=True, crop=True)
        frame = pickle.loads(base64.b64decode(frame_string.encode("utf-8")))

        et = np.round((time.time() - st), 3)
        if self.verbose:
            print(f"Image capture took {et} seconds.")

        # Use background if specified in this get_image() function, otherwise use the initialized value in the constructor
        if background is None:
            bg = self.bg
        else:
            bg = background
        
        # Subtract background - uniform for now
        frame = np.array(frame) - bg  # Frame after capture has type: list due to Pi transmission
        # Since there are negative values due to background subtraction, the array has d-type np.int32 (signed int).

        # Clip negative values to zero
        frame[frame < 0] = 0

        # Use the return_green_interp is specified in this function, use the initialized value otherwise
        if return_green_interp is None:
            rgi = self.return_green_interp
        else:
            rgi = return_green_interp

        # If required return the interpolated image
        if rgi:
            frame = return_interp_arr(frame, self.bayer_offsets)

        return self.transform(frame)  # Transform image if desired

    def flush(self, timeout_s=1e-3):
        """See :meth:`.Camera.flush`."""
        
        # Clears ungrabbed images from the queue

        time.sleep(timeout_s)
        self.cam.flush_raw()