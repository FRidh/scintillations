{ buildPythonPackage
, pytest
, cython
, numpy
, scipy
, streaming
}:

buildPythonPackage rec {
  name = "scintillations-${version}";
  version = "dev";

  src = ./.;

  buildInputs = [ pytest cython ];
  propagatedBuildInputs = [ numpy scipy streaming ];

  doCheck = false;
}
