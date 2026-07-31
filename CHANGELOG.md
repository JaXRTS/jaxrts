# Changelog

## [0.8.0] - 2026-07-31

This release most notably brings performance improvements for HNC and Born-Mermin ff calculations. This is achieved by using a better FFT implementation and using hard-coded algorithms to invert small matrices.
Further, we introduce a new way of defining instrument functions. While the old code is still working, the new `InstrumentFunction` class allows more transparent saving and can be extended easily.
We allow for summing of IPD models, and different shapes of bound-free (and free-bound) edges.

The ionization solver has been rewritten, removing code duplication, improving both performance and maintainability.

We also have created a quick-start jupyter notebook, that you can use online (with colab) to get stated without the need of installing anything on your machine.

### Features

- Create intro notebook([fdeaf6f](https://github.com/JaXRTS/jaxrts/commit/fdeaf6f7269e6b248f10ee805f9c2669411dae1a))
- IPDSum model([03cdef6](https://github.com/JaXRTS/jaxrts/commit/03cdef603a7e619824c76d0cb39c4e45bc40f229))
- Functions for getting all models, potentials and instrument functions([8055ff1a](https://github.com/JaXRTS/jaxrts/commit/8055ff1a08e0a9e0fe15463962ea2fcac23ea8ff))
- Models for the shape of bound-free edges([41e6b1](https://github.com/JaXRTS/jaxrts/commit/41e6b1670e676e9b5045fe778cb8c90092aa66fb))
- add BU solver([b5c7f4](https://github.com/JaXRTS/jaxrts/commit/b5c7f46e8b53955318761f1ff1caa38ef0fa772d))

### Bug Fixes

- Stabilize ITCFT against shifting of the energy grid([613daa3](https://github.com/JaXRTS/jaxrts/commit/613daa3ffd96a9a5cf745e3444d56e090de48ada))

### Other

- SiiNN save jaxrts version to .info file. Include sampling boundaries([eb43ef0](https://github.com/JaXRTS/jaxrts/commit/eb43ef0742d73266a6e60eed5eb1f2d40bf436dc),[20a8c2c](https://github.com/JaXRTS/jaxrts/commit/20a8c2c3703851d9c60de9e0a684629f85cb7ce9))

### Refactor

- Remove extensive code duplication in balance solver([0a9d6c0](https://github.com/JaXRTS/jaxrts/commit/0a9d6c0c641d94a9761492663e8b9f649ec686d3))
- Use IPDModel to collect models for IPDs([856f062](https://github.com/JaXRTS/jaxrts/commit/856f062c6df4b75e41c81d69eb5b47da0e66a67d))
- Create classes for InstrumentFunctions([67d3f6b](https://github.com/JaXRTS/jaxrts/commit/67d3f6b6f403b2ffb70141ef03d9eff01c0175b5))

### Documentation

- Show printing example, literature fixes([7a40aee](https://github.com/JaXRTS/jaxrts/commit/7a40aee7a74010948995ee89628f41d3ada6e10e))
- Add links to sidebar, provide better bibtex for paper([be79e12](https://github.com/JaXRTS/jaxrts/commit/be79e12156906db728296dd18a1047b0d664e2c4))
- Add info about loading and saving from disk([9e46ff2](https://github.com/JaXRTS/jaxrts/commit/9e46ff2763bd6798c98b8fddbba332f868725de4))
- New example: saha ionization map for CH plasma (#71)([c882388](https://github.com/JaXRTS/jaxrts/commit/c8823886564e84b9cb65f481fa87aeccc7aad1e1))
- Correct pip install command([94684fa](https://github.com/JaXRTS/jaxrts/commit/94684faaca63d344697e8e4ca45f3a79a71ba5ec), [510dcc8](https://github.com/JaXRTS/jaxrts/commit/510dcc8b4764beeed94efc888d333c5c61721ef7))
- Mention notebook in readme([00a9ad8](https://github.com/JaXRTS/jaxrts/commit/00a9ad8d19cae2e51097ccee7d6a1487b5039476))
- Improved docstrings([48a20fc](https://github.com/JaXRTS/jaxrts/commit/48a20fccb7995db38ad11ef0c55ea89386261fd1), [14a33d4](https://github.com/JaXRTS/jaxrts/commit/14a33d4b687a3b8f7bc580c2004c23dcddce15ad))

### Performance

- Shorter DST4, remove unused DST types([8d3ae96](https://github.com/JaXRTS/jaxrts/commit/8d3ae96133c86124f603befb848a615d1f060c90),[6dca5e7](https://github.com/JaXRTS/jaxrts/commit/6dca5e787d6052f2aeb5707642c0b01907c8baea),[5112ee6](https://github.com/JaXRTS/jaxrts/commit/5112ee69064d7c45b1a7a6573e7d6b656d4145f7))
- Don't re-iterate HNC per integration point in BM nu([3bdf9c7](https://github.com/JaXRTS/jaxrts/commit/3bdf9c7bfa09a6dc126518e662069f7273efc793))
- Hardcoded matrix inversion for low-size matrices([3938db1](https://github.com/JaXRTS/jaxrts/commit/3938db1d66a27306a45345207f9c8b64dbe5b0d8))
- Use solve over inv([af985ea](https://github.com/JaXRTS/jaxrts/commit/af985ea722ea29b82ad348e288b15a5c1a84d864))
- Simplify HNC matrix construction([90b07c5](https://github.com/JaXRTS/jaxrts/commit/90b07c5316872f69784ff39038a4ce63f60d61a4))

### Testing
- Unit tests for testing jax cache size([bbb6769](https://github.com/JaXRTS/jaxrts/commit/bb676929c3a3abadd5016ff784dc5030f6458a62)

### Miscellaneous Tasks

- Add requests to download_data extras([033af49](https://github.com/JaXRTS/jaxrts/commit/033af492cc7e1a74a829872e57e404e16cd421ab))
- Add citation file([2c7224](https://github.com/JaXRTS/jaxrts/commit/2c722479a9ec50220cc0d1d90f4272dbbe396156))
- Dependency upgrade (includes jax 0.10)([1210e9d](https://github.com/JaXRTS/jaxrts/commit/1210e9d96900f0209032b4c0d93dbc62926577a0))


**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.7.0...0.8.0


## [0.7.0] - 2026-04-16

This release brings smaller changes, including a renaming of the `saha` submodule to `ionization`, and a more natural default e-i Potential (now just a Coulomb potential). Further, it introduces a more stable interpolation of Sii with neural networks, a multi-species ionization solver in the TF model.

### Features

- Multi-species TF-AA ionization solver (#69)([4e4f90f](https://github.com/JaXRTS/jaxrts/commit/4e4f90fec13b021890628d602e1a941eb89718b6))

### Bug Fixes

- SiiNN interpolation works more robust with expanded integer ionization state (#66)([dd7fe0f](https://github.com/JaXRTS/jaxrts/commit/dd7fe0f84db16328a1d9c017c213d0d6635ff6f1))

### Refactor

- [**breaking**] Renaming saha.py to ionization.py (#68)([7c34f3e](https://github.com/JaXRTS/jaxrts/commit/7c34f3e5a04a270f7efdb13d7fe70d1f0c7de1da))
- [**breaking**] Changing the default electron-ion Potential to CoulombPotential([df6a494](https://github.com/JaXRTS/jaxrts/commit/df6a494e6aad97962378bed8fd3932518c29f188)

### Miscellaneous Tasks

- Update dependencies([3dcb429](https://github.com/JaXRTS/jaxrts/commit/3dcb429a22cd0751249f10197d11464d2f5bb832), [0c93035](https://github.com/JaXRTS/jaxrts/commit/0c93035cf60adc9459a1716deddfc7074dc53880), [5a3478c](https://github.com/JaXRTS/jaxrts/commit/5a3478ca46fd1b5a1973aee48a3e39ac3bc1b020))

**Full Changelog**: https://github.com/jaxrts/jaxrts/compare/0.6.0...0.7.0

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

