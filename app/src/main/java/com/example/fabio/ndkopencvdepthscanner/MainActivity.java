package com.example.fabio.ndkopencvdepthscanner;

import android.Manifest;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraManager;
import android.nfc.Tag;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.FloatProperty;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.Collection;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, SensorEventListener {

    private static final int MY_REQUEST_CODE = 666;
    private static final double K = 0.02;
    private static final float NS2S = 1.0f/1000000000.0f;


    private SensorManager mSensorManager;
    private Sensor mSensor;

    private static String TAG = "MainActivity";
    private static String TAG_CAMERA = "openCVCamera";

    JavaCameraView javaCameraView;
    Mat mRGBa1, mRGBa2, tmp, tmp1;
    boolean read1 = false, read2 = false, empty2 = true;
    double pts = 0;
    private double acc = 0, prevAcc = 0, speed = 0, displacement = 0;;
    ArrayList<Float> accReadings = new ArrayList<Float>(500), dtReadings = new ArrayList<Float>(5000);

    static{
        System.loadLibrary("MyOpencvLibs");
    }

    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS:
                    javaCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA},
                        MY_REQUEST_CODE);
            }
        }else{
            //Do Your Stuff
        }

        mSensorManager = (SensorManager) getSystemService(this.SENSOR_SERVICE);
        mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);

        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(View.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        javaCameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN){
                    if(!read1){
                        displacement = 0;
                        acc = 0;
                        prevAcc = 0;
                        pts = 0;
                        mSensorManager.registerListener(MainActivity.this, mSensor, SensorManager.SENSOR_DELAY_FASTEST);
                    }
                    read1 = true;
                }
                if (event.getAction() == MotionEvent.ACTION_UP){
                    mSensorManager.unregisterListener(MainActivity.this);
                    //Log.i(TAG, String.valueOf(displacement));
                    read2 = true;
                }
                return true;
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(javaCameraView!=null)
            javaCameraView.disableView();
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(javaCameraView!=null)
            javaCameraView.disableView();
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            //Log.i(TAG, "Opencv loaded successfully");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else {
            //Log.i(TAG, "Opencv not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        }
        mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_FASTEST);
        displacement = 0;
        acc = 0;
        prevAcc = 0;
        pts = 0;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRGBa1 = new Mat(height, width, CvType.CV_8UC4);
        mRGBa2 = new Mat(height, width, CvType.CV_8UC4);
        tmp = new Mat(height, width, CvType.CV_8UC4);
        tmp1 = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mRGBa1.release();
        mRGBa2.release();
        tmp.release();
        tmp1.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if(!read1) {
            tmp1.release();
            mRGBa1 = inputFrame.rgba();
            tmp1 = mRGBa1.clone();
            return tmp;
        }
        if(read1 && !read2){
            mRGBa2 = inputFrame.rgba();
            return mRGBa2;
        }
        tmp.release();
        tmp = mRGBa2.clone();
        //Log.i(TAG, String.valueOf(displacement));
        OpencvNativeClass.crossingsDetection(tmp1.getNativeObjAddr(), tmp.getNativeObjAddr());
        read1 = false;
        read2 = false;
        accReadings.clear();
        return tmp;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if(event.sensor.getType() != Sensor.TYPE_LINEAR_ACCELERATION)
            return;
        double ts = event.timestamp;
        double dt = (pts == 0 ? 0 : (ts - pts) * NS2S);
        pts = ts;
        acc = event.values[1] * K + prevAcc*(1 - K);
        prevAcc = acc;
        speed += acc * dt;
        displacement += speed * dt + 0.5 * acc * dt * dt;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    protected float lowPassRamp(float input, float output, float beta){
        if ( output == -9999 )
            return input;
        output = beta * output + (1 - beta) * input;
        return output;
    }

    public float dcBlocker(float unfiltered1, float unfiltered2, float filtered, float R){
        if (unfiltered2 == -9999f)
            return unfiltered1;
        return unfiltered1 - unfiltered2 + R*filtered;
    }
}
