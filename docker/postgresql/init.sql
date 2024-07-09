CREATE EXTENSION vector;
CREATE TYPE sparsevec AS (ndims int, indices int[], values float[]);
CREATE TYPE halfvec AS (ndims int, indices int[], values real[]);
