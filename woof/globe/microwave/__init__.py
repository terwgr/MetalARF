"""Microwave leg of the global assimilation design (item 5b): the ATMS
temperature-sounding channels as a clear-sky, over-ocean observation.

Modules
-------
``atms_fetch``
    One day of ATMS SDR granule pairs from the NOAA JPSS open-data
    buckets with the manifest (URL, size, SHA-256, per-hour volume,
    publication latency).  Orchestration only.
``atms_bridge``
    The Python side of the ``rw_atms`` Rust bridge: decode the IDPS HDF5
    granules to flat arrays, thin them onto latitude rings, read the
    results back as numpy.  No HDF5 is opened in Python.
``channels``
    The 22-channel ATMS plan: passbands, bandwidth, quasi-polarization,
    noise.
``absorption``
    Clear-air absorption, ITU-R P.676-13 Annex 1 (oxygen, water vapour,
    dry continuum) in the Rosenkranz line-mixing form.
``emissivity``
    Specular Fresnel ocean emissivity with the Meissner-Wentz (2004)
    dielectric constant.
``rte``
    The plane-parallel radiative transfer: column in, channel brightness
    temperatures out, Planck exact by default, Rayleigh-Jeans selectable;
    weighting functions for the calibration.
``columns``
    GDAS analysis columns (the pgrb2 0.25-degree isobaric product through
    the Rust mapped engine) interpolated in time to the observation
    hour and sampled at the thinned cells.
``score``
    The O-B scorecard: clear-sky over-ocean screening, per-channel bias
    and rmse before and after the linear bias correction, the diagnostics
    by scan angle, latitude band and 10 m wind.
``calibrate``
    The two-direction synthetic calibration the design demands before any
    real number is cited: the isothermal column, the planted warm layer,
    the null column, the Planck-versus-Rayleigh-Jeans term, and the
    comparison against an independent public absorption implementation.
``entry``
    The operator entry the scorecard promotes: the acceptance contract,
    the admitted channels with their errors, bias coefficients and
    vertical placement, the member operator that evaluates H(x) on every
    ensemble member, and the PointObs batches the filter takes.

Interface decisions recorded for the other lanes
-----------------------------------------------
* An ATMS observation for the filter is one thinned cell: mean
  brightness temperature per channel over the beams that fell into one
  grid cell in one time bin, with the beam count, the standard deviation
  across the beams (the representativeness read), the mean zenith and
  scan angle, and the mean time.  The thinning runs in Rust
  (``rw_atms thin``) against the caller's latitude rings and longitude
  count, so the same binary places beams on the Gaussian grid of any
  truncation or on the 0.25-degree analysis grid.
* The forward operator takes a :class:`rte.Column` batch (shared pressure
  levels, per-column temperature, specific humidity, surface pressure,
  skin temperature, optional 2 m temperature) and the viewing geometry,
  and returns channel brightness temperatures.  For the spectral model
  the caller synthesizes the column on the Gaussian grid; the operator
  does not read a checkpoint itself.
* The observation error the operator entry declares per channel is the
  instrument NEdT combined with the operator's calibrated clear-sky
  residual after bias correction (the ``rmse_after`` column of the O-B
  scorecard), never the NEdT alone.
* The bias correction is per channel and linear in what the member
  operator can evaluate for a row: ``a + b (B - mean_B) + c (sec z - 1)``
  with B the member's own brightness temperature and z the cell's
  zenith angle, fitted on one half of a day's clear-sky over-ocean cells
  in time order and scored on the other half; its coefficients are part
  of the operator entry and carry the day they were fitted on.  The
  analysis 10 m wind is a diagnostic predictor only: a channel that
  passes the bar only with it is surface-limited and is not admitted.
* A channel is admitted on its own reading: the entry lists the sounding
  channels whose geometry-corrected rmse is inside the bar and names the
  nearest term for each refused one.  The filter offers one PointObs
  batch per admitted channel, rows keyed by cell identity
  (``atms:<satellite>:<bin>:<ring>:<lon>:chNN``), localised in model space
  through the channel's weighting-function profile with a vertical length
  measured on the ensemble at every window (``entry.channel_profile``,
  ``da.localisation``); the centroid in ln p names the row for thinning.
* The stream is the door's ``atms`` stream (``radiance_streams``): the
  window's granules through ``rw_atms``, the cells screened against the
  control background and the observations, the slope and scan terms of the
  bias correction moved by the window's own departures (the constant stays
  the entry's, the anchor); whether it is in the default set is the
  observation scorecard's verdict on the door page.
"""
