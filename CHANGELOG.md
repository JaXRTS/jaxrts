# Changelog

## [0.6.0] - 2026-03-10

This release abbreviates the LFC model names and rewrite NNSiiModels, neglects
a free-bound contribution by default, adds the Charged Hard Sphere model for
ionic scattering, and automatically deploys the documentation to
https://jaxrts.github.io/jaxrts.

### Breaking changes
- [**breaking**] Rename LFC models([14264d0](https://github.com/JaXRTS/jaxrts/commit/14264d06eb76ae0bd22350b0aaedf9af2b21b508))
- Set default free-bond model to Neglect([aeef1f4](https://github.com/JaXRTS/jaxrts/commit/aeef1f41d0ebe553b83ddfcde6f866190f4881ca))

### Features

- Add function to calculate density from electron-density for mixed systems (#53)([9db90df](https://github.com/JaXRTS/jaxrts/commit/9db90df5f656d8bc66430bf27101c3c72dfc0bd9))
- Static structure factors in CHS from Gregori.2007([113ed42](https://github.com/JaXRTS/jaxrts/commit/113ed424b5d84b55282a214879e798280f1fa314))
- Test LFC model validity([7939497](https://github.com/JaXRTS/jaxrts/commit/7939497b622811f8c24456b18dfe5d1dd1da5805))
- Add Gregori2007 and Farid1993 sLFC models([5212cfa](https://github.com/JaXRTS/jaxrts/commit/5212cfad9fe5ae5204b4b00249cfd1e09f506b37))
- Adding the analytical fit of More for Z_f in plasma_physics (#60)([43b224b](https://github.com/JaXRTS/jaxrts/commit/43b224b84ee59911f14e775bd16cc023b08552e3))
- Gregori.2006b extension of Arkhipov models to multicomponent (#56)([1fded18](https://github.com/JaXRTS/jaxrts/commit/1fded1814ac8f4043a40f93f988e93deb90844c1))

### Bug Fixes

- Have all LFC functions return Quantities([d1d1213](https://github.com/JaXRTS/jaxrts/commit/d1d121393ccd6a34aa4a73020e9a3407122e6614))
- `ConstantIPD` now works with multiple entries for different plasma constituents([9bfc5a2](https://github.com/JaXRTS/jaxrts/commit/9bfc5a27f06f4171ff5dbce36e529349ca1766a7))
- Set `B` to unity in bound-free models (#54)([4ded46d](https://github.com/JaXRTS/jaxrts/commit/4ded46d045827e916ed29ab1288b6431530112ba))
- Temperature average after calculation of effective temperature in `Gregori2006IonFeat` (#56)([93e5e78](https://github.com/JaXRTS/jaxrts/commit/93e5e78934f3c89135e586b8bb34b0770ac04925))

### Refactor

- [**breaking**] Unify NNSiiModel class (#62)([9230b75](https://github.com/JaXRTS/jaxrts/commit/9230b7566693d0eeb4924bde01d34d851c977032))

### Documentation

- Create workflow for automatically generating the documentation([bf8dda9](https://github.com/JaXRTS/jaxrts/commit/bf8dda9b622c85a0610e1c752120ea3262e83e78))
- Better documentation of the LFC models currently implemented([530b762](https://github.com/JaXRTS/jaxrts/commit/530b762a32a3c044eb510f453aff3fa426af69af))
- Improving docstring of SommerfeldChemPot Model([ad75381](https://github.com/JaXRTS/jaxrts/commit/ad7538127222c5956bae7c9465602fd254c72f8c))
- Change citekeys in the documentation to author-year([3466c5c](https://github.com/JaXRTS/jaxrts/commit/3466c5c9b138b9d7ce3b6508dbd6cf54d8f5f353))

### Performance

- Extend upper bound of CHS root-finding([e948fa7](https://github.com/JaXRTS/jaxrts/commit/e948fa77965f5fbae53842070b732af8f09dfe02))

### Testing

- Test result of Dornheim interpolation([cac8d6a](https://github.com/JaXRTS/jaxrts/commit/cac8d6a013faa80668f8ba84526d0a5241eb022d))
- Fix saved state, as the new default moved order of models([482c631](https://github.com/JaXRTS/jaxrts/commit/482c631f481c6467a2b7db97ca1caf478e6f4e52))
- Fix data not found; fix spelling of author([bdfc12d](https://github.com/JaXRTS/jaxrts/commit/bdfc12d1256eba1f50a7855a45a38b658dbf51c5))

### Miscellaneous Tasks

- Variant of the logo with bright text([e72125f](https://github.com/JaXRTS/jaxrts/commit/e72125f1146d2d1751592e0dfe2ba53b9f6961a9))
- Set ruff line-length to 79([a8bc5ea](https://github.com/JaXRTS/jaxrts/commit/a8bc5eaac21d2bd9d4c8208fc4033c0ab8a3c068))
- Move bisection from saha to helpers([c48e93d](https://github.com/JaXRTS/jaxrts/commit/c48e93de682e29ce3c1c35aa587dd4f89f3e9d78))
- Update dependencies([c631222](https://github.com/JaXRTS/jaxrts/commit/c63122231b3009eec01ee714eec2dd8bf9c85a2a))
- Remove unnecessary .drone file([a211a21](https://github.com/JaXRTS/jaxrts/commit/a211a21d0e64e8a20f52b015ed09888ccc8d9176))

**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.5.0...0.6.0

## [0.5.0] - 2025-10-30

### Features

- Logo (#49)([6f3ec17](https://github.com/JaXRTS/jaxrts/commit/6f3ec17c9b819e07900a6da3c3a57d155abe6918))
- `print` options for PlasmaStates, Models, ect. (#47)([5412c929](https://github.com/JaXRTS/jaxrts/commit/5412c9295b89b748fc23fa944e663e6a42fa27b1))

### Bug Fixes

- Logo: Convert text to path (#50)([f19ce5b](https://github.com/JaXRTS/jaxrts/commit/f19ce5b6d3251555bee5182fe05e8e10f18f0fef))

### Documentation

- Improve sphinx documentation([9fd3cbb](https://github.com/JaXRTS/jaxrts/commit/9fd3cbbd3ae18ee09f358608b95a36c970c0ba7f))

### Testing

- Avoid opening tempfiles several times to have working test on Windows([382a8f8](https://github.com/JaXRTS/jaxrts/commit/382a8f8ffe9df92144c9a06494845593a77f4812))

### Miscellaneous Tasks

- Move from poetry optional dependencies to `[project.optional-dependencies]`([da7959f](https://github.com/JaXRTS/jaxrts/commit/da7959f5a06a249cb6e5d098e23e2813c323e716))


**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.4.1...0.5.0


## [0.4.1] - 2025-10-24

### Bug Fixes

- ElectronicLFCDornheimAnalyticalInterp not evaluating at full k ([42a6e59](https://github.com/JaXRTS/jaxrts/commit/42a6e59a0953729a949b39e24af5b093a832907f))

**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.4.0...0.4.1

## [0.4.0] - 2025-10-23

### Breaking

- Rename `RPA` and `BornMermin_Full` models ([e8bd2eb](https://github.com/JaXRTS/jaxrts/commit/e8bd2eb81cbe47a2fdccc991b5ee0dc634b094ca))

### Features

- Add FormfactorLowering model (#27)([07be65a](https://github.com/JaXRTS/jaxrts/commit/07be65ae8e2f94b1231a69232e0322c212fb7ef4))
- Multicomponent SVT-HNC (#39)([6b8f278](https://github.com/JaXRTS/jaxrts/commit/6b8f278bc761517ff738a313687ba262f889ccf2))
- citation method for models and hnc_potentials (#38)([d64a645](https://github.com/JaXRTS/jaxrts/commit/d64a645cbde49df9d4aaf0ff7f45aff7e1dd3cc3)]
- PlasmaParameter calculator (#41)([4e5bed7](https://github.com/JaXRTS/jaxrts/commit/4e5bed79e738da46c1f53d85c18d3310218949bb))
- Sommerfeld Chemical Potential ([4e5bed7](https://github.com/JaXRTS/jaxrts/commit/4e5bed79e738da46c1f53d85c18d3310218949bb))
- Allow to build instrument-function from array([529f8d6](https://github.com/JaXRTS/jaxrts/commit/529f8d6dd6e8d3e69dcd2d0c640cc594b17b53d3))
- Allow to load instrument functions with different units for x([6c7f36d](https://github.com/JaXRTS/jaxrts/commit/6c7f36db9faf835f80bc7bb48461dbc26fbae3e4))
- Allow ion_feature.free_electron_susceptilibily_RPA to use an lfc([652bac9](https://github.com/JaXRTS/jaxrts/commit/652bac98a27d3ce5f93970c5c6591d93a9a07934))

### Bug Fixes

- Saha-Boltzmann solver with IPD (#42)([e708926](https://github.com/JaXRTS/jaxrts/commit/e7089260882251e4d2fb0bdf3c5e54898f3d6675))
- Set default for ThreePotentialHNCIonFeat to Coulomb for ii([370537f](https://github.com/JaXRTS/jaxrts/commit/370537f91e855b81d142b94857f304613e2d6a4e))
- Return type of instrument_from_file is callable([e42eb49](https://github.com/JaXRTS/jaxrts/commit/e42eb49be7f3d6f5c630a94f54aec8927385cfce))
- Units on supergaussian instrument function([2440f8a](https://github.com/JaXRTS/jaxrts/commit/2440f8a2a9d9e5d0ec36e88bda55179539061bec))
- [SiiNN] ShapeDtypeStruct immutable([524efd8](https://github.com/JaXRTS/jaxrts/commit/524efd88645a07aabd845ee82090fa4507df3698))

### Documentation

- Many improvements to docstrings, added cite keys

### Testing

- Make test_peak_position_stability_with_convolution a proper test([e099236](https://github.com/JaXRTS/jaxrts/commit/e09923669291d9d4e190f124e9e0ffb71f35426b))
- Re-establish test for integer expansion & W_r calculation([3a7f747](https://github.com/JaXRTS/jaxrts/commit/3a7f74784a50817eb268f74ccc662a215ef55805))
- Test IPD models against Lin.2017

### Miscellaneous Tasks
- Dependency updates, supporting jax < 0.8.0

**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.3.0...0.4.0

