package org.tensorflow.lite.examples.detection.barcode;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class BarcodeRequest {

    @SerializedName("qrcodes")
    @Expose
    public String qrcodes;

    public BarcodeRequest(String qrcodes){
        this.qrcodes = qrcodes;
    }

    public String getQrcodes() {
        return qrcodes;
    }

    public void setQrcodes(String qrcodes) {
        this.qrcodes = qrcodes;
    }
}
