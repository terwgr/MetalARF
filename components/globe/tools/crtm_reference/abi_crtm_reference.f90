! abi_crtm_reference: CRTM (v3.1.1) as the numerical reference forward model for the ABI
! infrared operator of Arwen Global.
!
! Reads a column file of schema gpuwm-da.abi-columns.v2 (or v1; little-endian stream, written by
! arwen_global.abi_reference.write_columns), runs the CRTM K-matrix model for the
! requested ABI channels on every column, and writes a gpuwm-da.abi-crtm.v1 stream: the
! brightness temperature, the surface emissivity CRTM used, the radiance, the skin-temperature
! Jacobian, the layer temperature and water-vapor Jacobians, and the layer optical depths.
!
! This program is a reference instrument, not a data path: the shipped forward operator is
! Rust; this measures what the reference says on the same columns so the Rust operator's error
! can be named by term (absorption, source function, surface emissivity) instead of guessed.
!
! Usage:
!   abi_crtm_reference SENSOR_ID COEFF_DIR COLUMNS.bin OUT.bin EMIS_MODE IRLAND_FILE CH1 [CH2 ...]
!     SENSOR_ID    CRTM sensor id (abi_g16, abi_g18); the SpcCoeff and TauCoeff must be in COEFF_DIR
!     COEFF_DIR    a flat directory (trailing slash) holding every coefficient file CRTM_Init loads
!     EMIS_MODE    0: CRTM's own surface models (IR water by wind speed, IR land by the class table)
!                  1: the per-column, per-channel emissivity carried in the columns file
!     IRLAND_FILE  the IR land emissivity table (NPOESS.IRland.EmisCoeff.bin or IGBP.IRland.EmisCoeff.bin)
!                  whose type index the columns file's land_type follows
!     CHn          ABI channel numbers (8, 13)
PROGRAM abi_crtm_reference
  USE CRTM_Module
  IMPLICIT NONE

  CHARACTER(*), PARAMETER :: PROGRAM_NAME = 'abi_crtm_reference'
  INTEGER, PARAMETER :: COLUMNS_MAGIC = 1128874561   ! "ABIC" as a little-endian int32
  INTEGER, PARAMETER :: OUTPUT_MAGIC  = 1380532801   ! "ABIR" as a little-endian int32
  INTEGER, PARAMETER :: N_ABSORBERS = 2, N_CLOUDS = 0, N_AEROSOLS = 0, N_SENSORS = 1
  INTEGER, PARAMETER :: SEA_WATER_TYPE = 1, FRESH_SNOW_TYPE = 2, FRESH_ICE_TYPE = 1
  REAL(fp), PARAMETER :: SNOW_COVER_DEPTH_M = 0.01_fp

  CHARACTER(256) :: sensor_id, coeff_dir, columns_file, out_file, irland_file, arg, version, message
  INTEGER :: emis_mode, n_request, n_args, i, m, l, k, err, alloc_stat
  INTEGER :: uin, uout
  INTEGER, ALLOCATABLE :: request(:)
  ! columns file
  INTEGER :: magic, file_version, n_prof, n_layers, n_user_chan
  INTEGER, ALLOCATABLE :: user_chan(:), climatology(:), land_type(:)
  REAL(fp), ALLOCATABLE :: lat(:), lon(:), zenith(:), land_frac(:), skin_k(:), psfc_hpa(:), &
                           wind10(:), snowh_m(:), seaice(:), plev(:,:), play(:,:), tk(:,:), &
                           q_gkg(:,:), o3_ppmv(:,:), emis_user(:,:), a_half(:), b_half(:)
  ! CRTM structures (one profile at a time)
  TYPE(CRTM_ChannelInfo_type) :: chinfo(N_SENSORS)
  TYPE(CRTM_Geometry_type)    :: geo(1)
  TYPE(CRTM_Options_type)     :: opt(1)
  TYPE(CRTM_Atmosphere_type)  :: atm(1)
  TYPE(CRTM_Surface_type)     :: sfc(1)
  TYPE(CRTM_RTSolution_type), ALLOCATABLE :: rts(:,:), rts_k(:,:)
  TYPE(CRTM_Atmosphere_type), ALLOCATABLE :: atm_k(:,:)
  TYPE(CRTM_Surface_type),    ALLOCATABLE :: sfc_k(:,:)
  INTEGER :: n_chan
  INTEGER, ALLOCATABLE :: chan_list(:)
  REAL(fp), ALLOCATABLE :: emis_row(:)
  ! outputs (channel-major on disk: [chan][prof][layer] in C order = (layer, prof, chan) in Fortran)
  REAL(fp), ALLOCATABLE :: bt(:,:), emis(:,:), rad(:,:), jac_tskin(:,:), sfc_planck(:,:), &
                           jac_t(:,:,:), jac_q(:,:,:), od(:,:,:)
  REAL(fp) :: water_frac, snow_frac, ice_frac, land_cov
  INTEGER :: n_bad, user_index

  n_args = COMMAND_ARGUMENT_COUNT()
  IF ( n_args < 7 ) THEN
    WRITE(*,'(a)') 'usage: abi_crtm_reference SENSOR_ID COEFF_DIR COLUMNS.bin OUT.bin EMIS_MODE IRLAND_FILE CH1 [CH2 ...]'
    STOP 2
  END IF
  CALL GET_COMMAND_ARGUMENT(1, sensor_id)
  CALL GET_COMMAND_ARGUMENT(2, coeff_dir)
  CALL GET_COMMAND_ARGUMENT(3, columns_file)
  CALL GET_COMMAND_ARGUMENT(4, out_file)
  CALL GET_COMMAND_ARGUMENT(5, arg); READ(arg,*) emis_mode
  CALL GET_COMMAND_ARGUMENT(6, irland_file)
  n_request = n_args - 6
  ALLOCATE(request(n_request))
  DO i = 1, n_request
    CALL GET_COMMAND_ARGUMENT(6 + i, arg); READ(arg,*) request(i)
  END DO

  ! ---- the columns file ------------------------------------------------------------------
  OPEN(NEWUNIT=uin, FILE=TRIM(columns_file), ACCESS='stream', FORM='unformatted', STATUS='old', ACTION='read', IOSTAT=err)
  IF ( err /= 0 ) THEN
    WRITE(*,'(a)') 'cannot open '//TRIM(columns_file); STOP 3
  END IF
  READ(uin) magic, file_version, n_prof, n_layers, n_user_chan
  IF ( magic /= COLUMNS_MAGIC .OR. ( file_version /= 1 .AND. file_version /= 2 ) ) THEN
    WRITE(*,'(a,i0,a,i0)') 'not a gpuwm-da.abi-columns v1 or v2 file: magic ', magic, ' version ', file_version; STOP 3
  END IF
  ALLOCATE(user_chan(MAX(n_user_chan,1)), climatology(n_prof), land_type(n_prof), lat(n_prof), lon(n_prof), &
           zenith(n_prof), land_frac(n_prof), skin_k(n_prof), psfc_hpa(n_prof), wind10(n_prof), snowh_m(n_prof), &
           seaice(n_prof), plev(0:n_layers, n_prof), play(n_layers, n_prof), tk(n_layers, n_prof), &
           q_gkg(n_layers, n_prof), o3_ppmv(n_layers, n_prof), emis_user(MAX(n_user_chan,1), n_prof))
  ALLOCATE(a_half(0:n_layers), b_half(0:n_layers))
  a_half = 0.0_fp; b_half = 0.0_fp
  IF ( file_version == 2 ) READ(uin) a_half, b_half      ! the model's vertical coordinate (v2)
  IF ( n_user_chan > 0 ) READ(uin) user_chan(1:n_user_chan)
  READ(uin) lat, lon, zenith, land_frac, skin_k, psfc_hpa, wind10, snowh_m, seaice
  READ(uin) climatology, land_type
  READ(uin) plev, play, tk, q_gkg, o3_ppmv
  IF ( n_user_chan > 0 ) READ(uin) emis_user(1:n_user_chan, :)
  CLOSE(uin)
  WRITE(*,'(a,i0,a,i0,a)') 'columns: ', n_prof, ' profiles of ', n_layers, ' layers'

  ! ---- CRTM ------------------------------------------------------------------------------
  CALL CRTM_Version(version)
  WRITE(*,'(a,a)') 'CRTM version ', TRIM(version)
  err = CRTM_Init( (/sensor_id/), chinfo, File_Path=TRIM(coeff_dir), IRlandCoeff_File=TRIM(irland_file), &
                   Load_CloudCoeff=.FALSE., Load_AerosolCoeff=.FALSE., Quiet=.TRUE. )
  IF ( err /= SUCCESS ) THEN
    WRITE(*,'(a)') 'CRTM_Init failed'; STOP 4
  END IF
  err = CRTM_ChannelInfo_Subset( chinfo(1), Channel_Subset=request )
  IF ( err /= SUCCESS ) THEN
    WRITE(*,'(a)') 'CRTM_ChannelInfo_Subset failed (is every requested channel a sensor channel?)'; STOP 4
  END IF
  n_chan = SUM(CRTM_ChannelInfo_n_Channels(chinfo))
  ALLOCATE(chan_list(n_chan))
  chan_list = PACK(chinfo(1)%Sensor_Channel, chinfo(1)%Process_Channel)
  WRITE(*,'(a,i0,a,*(1x,i0))') 'sensor ', n_chan, ' channels selected:', chan_list

  ALLOCATE(rts(n_chan,1), rts_k(n_chan,1), atm_k(n_chan,1), sfc_k(n_chan,1), emis_row(n_chan), STAT=alloc_stat)
  ALLOCATE(bt(n_prof,n_chan), emis(n_prof,n_chan), rad(n_prof,n_chan), jac_tskin(n_prof,n_chan), &
           sfc_planck(n_prof,n_chan), jac_t(n_layers,n_prof,n_chan), jac_q(n_layers,n_prof,n_chan), &
           od(n_layers,n_prof,n_chan), STAT=alloc_stat)
  CALL CRTM_Atmosphere_Create( atm, n_layers, N_ABSORBERS, N_CLOUDS, N_AEROSOLS )
  CALL CRTM_Atmosphere_Create( atm_k, n_layers, N_ABSORBERS, N_CLOUDS, N_AEROSOLS )
  CALL CRTM_RTSolution_Create( rts, n_layers )
  CALL CRTM_RTSolution_Create( rts_k, n_layers )
  CALL CRTM_Options_Create( opt, n_chan )
  IF ( .NOT. CRTM_Options_Associated(opt(1)) ) THEN
    WRITE(*,'(a)') 'options allocation failed'; STOP 4
  END IF

  n_bad = 0
  DO m = 1, n_prof
    ! atmosphere, top of atmosphere first
    atm(1)%Climatology = climatology(m)
    atm(1)%Absorber_Id(1:2) = (/ H2O_ID, O3_ID /)
    atm(1)%Absorber_Units(1:2) = (/ MASS_MIXING_RATIO_UNITS, VOLUME_MIXING_RATIO_UNITS /)
    atm(1)%Level_Pressure(0:n_layers) = plev(0:n_layers, m)
    atm(1)%Pressure(1:n_layers) = play(:, m)
    atm(1)%Temperature(1:n_layers) = tk(:, m)
    atm(1)%Absorber(1:n_layers, 1) = q_gkg(:, m)
    atm(1)%Absorber(1:n_layers, 2) = o3_ppmv(:, m)
    ! surface: land, water, sea ice on the water part, snow on the land part above 1 cm depth
    land_cov = MIN(MAX(land_frac(m), 0.0_fp), 1.0_fp)
    water_frac = 1.0_fp - land_cov
    ice_frac = MIN(MAX(seaice(m), 0.0_fp), 1.0_fp) * water_frac
    water_frac = water_frac - ice_frac
    snow_frac = 0.0_fp
    IF ( snowh_m(m) > SNOW_COVER_DEPTH_M .OR. land_type(m) == 15 ) THEN
      snow_frac = land_cov
      land_cov = 0.0_fp
    END IF
    CALL CRTM_Surface_Zero( sfc )
    sfc(1)%Land_Coverage = land_cov
    sfc(1)%Water_Coverage = water_frac
    sfc(1)%Snow_Coverage = snow_frac
    sfc(1)%Ice_Coverage = ice_frac
    sfc(1)%Land_Type = MAX(land_type(m), 1)
    IF ( land_type(m) == 17 ) sfc(1)%Land_Type = 10
    sfc(1)%Water_Type = SEA_WATER_TYPE
    sfc(1)%Snow_Type = FRESH_SNOW_TYPE
    sfc(1)%Ice_Type = FRESH_ICE_TYPE
    sfc(1)%Land_Temperature = skin_k(m)
    sfc(1)%Water_Temperature = skin_k(m)
    sfc(1)%Snow_Temperature = skin_k(m)
    sfc(1)%Ice_Temperature = skin_k(m)
    sfc(1)%Wind_Speed = wind10(m)
    sfc(1)%Snow_Depth = snowh_m(m) * 1000.0_fp
    ! geometry: the block's mean satellite zenith; no sun (thermal channels only)
    CALL CRTM_Geometry_SetValue( geo, Sensor_Zenith_Angle=zenith(m), Source_Zenith_Angle=100.0_fp )
    ! options: user emissivity when asked
    opt(1)%Use_Emissivity = .FALSE.
    IF ( emis_mode == 1 ) THEN
      DO l = 1, n_chan
        user_index = 0
        DO k = 1, n_user_chan
          IF ( user_chan(k) == chan_list(l) ) user_index = k
        END DO
        IF ( user_index == 0 ) THEN
          WRITE(*,'(a,i0)') 'no user emissivity for channel ', chan_list(l); STOP 5
        END IF
        emis_row(l) = emis_user(user_index, m)
      END DO
      opt(1)%Use_Emissivity = .TRUE.
      CALL CRTM_Options_SetEmissivity( opt(1), emis_row )
    END IF
    ! K-matrix inputs: dTb/dx
    CALL CRTM_Atmosphere_Zero( atm_k )
    CALL CRTM_Surface_Zero( sfc_k )
    rts_k(:,1)%Radiance = ZERO
    rts_k(:,1)%Brightness_Temperature = ONE
    err = CRTM_K_Matrix( atm, sfc, rts_k, geo, chinfo, atm_k, sfc_k, rts, Options=opt )
    IF ( err /= SUCCESS ) THEN
      n_bad = n_bad + 1
      bt(m,:) = -999.0_fp; emis(m,:) = -999.0_fp; rad(m,:) = -999.0_fp; jac_tskin(m,:) = -999.0_fp
      sfc_planck(m,:) = -999.0_fp; jac_t(:,m,:) = -999.0_fp; jac_q(:,m,:) = -999.0_fp; od(:,m,:) = -999.0_fp
      CYCLE
    END IF
    DO l = 1, n_chan
      bt(m,l) = rts(l,1)%Brightness_Temperature
      emis(m,l) = rts(l,1)%Surface_Emissivity
      rad(m,l) = rts(l,1)%Radiance
      sfc_planck(m,l) = rts(l,1)%Surface_Planck_Radiance
      jac_tskin(m,l) = sfc_k(l,1)%Land_Temperature + sfc_k(l,1)%Water_Temperature + &
                       sfc_k(l,1)%Snow_Temperature + sfc_k(l,1)%Ice_Temperature
      jac_t(:,m,l) = atm_k(l,1)%Temperature(1:n_layers)
      jac_q(:,m,l) = atm_k(l,1)%Absorber(1:n_layers, 1)
      od(:,m,l) = rts(l,1)%Layer_Optical_Depth(1:n_layers)
    END DO
    IF ( MOD(m, 2000) == 0 ) WRITE(*,'(a,i0,a,i0)') 'done ', m, ' of ', n_prof
  END DO
  WRITE(*,'(a,i0,a)') 'profiles CRTM refused: ', n_bad, ' (written as -999)'

  ! ---- output ----------------------------------------------------------------------------
  OPEN(NEWUNIT=uout, FILE=TRIM(out_file), ACCESS='stream', FORM='unformatted', STATUS='replace', ACTION='write')
  WRITE(uout) OUTPUT_MAGIC, 1, n_prof, n_layers, n_chan, emis_mode, n_bad
  WRITE(uout) chan_list
  WRITE(uout) bt, emis, rad, jac_tskin, sfc_planck   ! each (n_prof, n_chan) column-major = [chan][prof]
  WRITE(uout) jac_t, jac_q, od                       ! each (n_layers, n_prof, n_chan) = [chan][prof][layer]
  CLOSE(uout)

  err = CRTM_Destroy( chinfo )
  CALL CRTM_Atmosphere_Destroy( atm )
  CALL CRTM_Atmosphere_Destroy( atm_k )
  CALL CRTM_Options_Destroy( opt )
  WRITE(*,'(a)') 'wrote '//TRIM(out_file)
END PROGRAM abi_crtm_reference
