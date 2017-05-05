package com.example.fabio.ndkopencvdepthscanner;

/**
 * Created by Fabio on 17/04/2017.
 */

public class OpencvNativeClass {
    public native static void crossingsDetection(long addrRgba1, long addrRgba2);
}
