# Changelog

## [0.5.0] - 2025-10-30

### Features

- Logo (#49)([6f3ec17](6f3ec17c9b819e07900a6da3c3a57d155abe6918))
- `print` options for PlasmaStates, Models, ect. (#47)([5412c929](5412c9295b89b748fc23fa944e663e6a42fa27b1))

### Bug Fixes

- Logo: Convert text to path (#50)([f19ce5b](f19ce5b6d3251555bee5182fe05e8e10f18f0fef))

### Documentation

- Improve sphinx documentation([9fd3cbb](9fd3cbbd3ae18ee09f358608b95a36c970c0ba7f))

### Testing

- Avoid opening tempfiles several times to have working test on Windows([382a8f8](382a8f8ffe9df92144c9a06494845593a77f4812))

### Miscellaneous Tasks

- Move from poetry optional dependencies to `[project.optional-dependencies]`([da7959f](da7959f5a06a249cb6e5d098e23e2813c323e716))


**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.4.1...0.5.0


## [0.4.1] - 2025-10-24

### Bug Fixes

- ElectronicLFCDornheimAnalyticalInterp not evaluating at full k ([42a6e59](42a6e59a0953729a949b39e24af5b093a832907f))

**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.4.0...0.4.1

## [0.4.0] - 2025-10-23

### Breaking

- Rename `RPA` and `BornMermin_Full` models ([e8bd2eb](e8bd2eb81cbe47a2fdccc991b5ee0dc634b094ca))

### Features

- Add FormfactorLowering model (#27)([07be65a](07be65ae8e2f94b1231a69232e0322c212fb7ef4))
- Multicomponent SVT-HNC (#39)([6b8f278](6b8f278bc761517ff738a313687ba262f889ccf2))
- citation method for models and hnc_potentials (#38)([d64a645](d64a645cbde49df9d4aaf0ff7f45aff7e1dd3cc3)]
- PlasmaParameter calculator (#41)([4e5bed7](4e5bed79e738da46c1f53d85c18d3310218949bb))
- Sommerfeld Chemical Potential ([4e5bed7](4e5bed79e738da46c1f53d85c18d3310218949bb))
- Allow to build instrument-function from array([529f8d6](529f8d6dd6e8d3e69dcd2d0c640cc594b17b53d3))
- Allow to load instrument functions with different units for x([6c7f36d](6c7f36db9faf835f80bc7bb48461dbc26fbae3e4))
- Allow ion_feature.free_electron_susceptilibily_RPA to use an lfc([652bac9](652bac98a27d3ce5f93970c5c6591d93a9a07934))

### Bug Fixes

- Saha-Boltzmann solver with IPD (#42)([e708926](e7089260882251e4d2fb0bdf3c5e54898f3d6675))
- Set default for ThreePotentialHNCIonFeat to Coulomb for ii([370537f](370537f91e855b81d142b94857f304613e2d6a4e))
- Return type of instrument_from_file is callable([e42eb49](e42eb49be7f3d6f5c630a94f54aec8927385cfce))
- Units on supergaussian instrument function([2440f8a](2440f8a2a9d9e5d0ec36e88bda55179539061bec))
- [SiiNN] ShapeDtypeStruct immutable([524efd8](524efd88645a07aabd845ee82090fa4507df3698))

### Documentation

- Many improvements to docstrings, added cite keys

### Testing

- Make test_peak_position_stability_with_convolution a proper test([e099236](e09923669291d9d4e190f124e9e0ffb71f35426b))
- Re-establish test for integer expansion & W_r calculation([3a7f747](3a7f74784a50817eb268f74ccc662a215ef55805))
- Test IPD models against Lin.2017

### Miscellaneous Tasks
- Dependency updates, supporting jax < 0.8.0

**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.3.0...0.4.0

