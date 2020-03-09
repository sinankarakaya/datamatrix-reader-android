package org.tensorflow.lite.examples.detection.barcode;


import android.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.tensorflow.lite.examples.detection.CameraActivity;
import org.tensorflow.lite.examples.detection.R;

public class BarcodeListFragment extends Fragment {

    private View parentView;
    private RecyclerView recyclerView;
    private AppCompatActivity parentActivity;
    private BarcodeAdapter adapter;
    private static BarcodeListFragment fragment;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        parentView =  inflater.inflate(R.layout.tfe_od_barcode_fragment,container,false);
        parentActivity  = CameraActivity.cameraActivity;
        fragment = this;
        this.setupViews();
        return parentView;
    }

    public void setupViews(){
        RecyclerView.LayoutManager mLayoutManager = new LinearLayoutManager(parentActivity.getApplicationContext());
        recyclerView = parentView.findViewById(R.id.barcode_view);
        recyclerView.setLayoutManager(mLayoutManager);
        recyclerView.setItemAnimator(new DefaultItemAnimator());
        adapter = new BarcodeAdapter(parentActivity.getApplicationContext(),CameraActivity.getBarcode());
        recyclerView.setAdapter(adapter);
        recyclerView.addItemDecoration(new DividerItemDecoration(recyclerView.getContext(), DividerItemDecoration.VERTICAL));

    }

    public static void updateAdapter(){
        if(fragment!=null)
            fragment.adapter.notifyDataSetChanged();
    }


}
