package org.tensorflow.lite.examples.detection.barcode;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class BarcodeResponse {

    @SerializedName("success")
    @Expose
    public String success;

    @SerializedName("message")
    @Expose
    public String message;



}
