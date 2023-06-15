var wms_layers = [];


        var lyr_GoogleMaps_0 = new ol.layer.Tile({
            'title': 'Google Maps',
            'type': 'base',
            'opacity': 1.000000,
            
            
            source: new ol.source.XYZ({
    attributions: ' ',
                url: 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
            })
        });
var format_disponibilidad_1 = new ol.format.GeoJSON();
var features_disponibilidad_1 = format_disponibilidad_1.readFeatures(json_disponibilidad_1, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_disponibilidad_1 = new ol.source.Vector({
    attributions: ' ',
});
jsonSource_disponibilidad_1.addFeatures(features_disponibilidad_1);
var lyr_disponibilidad_1 = new ol.layer.Vector({
                declutter: true,
                source:jsonSource_disponibilidad_1, 
                style: style_disponibilidad_1,
                interactive: true,
    title: 'disponibilidad<br />\
    <img src="styles/legend/disponibilidad_1_0.png" /> 0<br />\
    <img src="styles/legend/disponibilidad_1_1.png" /> 1 - 6<br />\
    <img src="styles/legend/disponibilidad_1_2.png" /> 6 - 10<br />\
    <img src="styles/legend/disponibilidad_1_3.png" /> 10 - 14<br />\
    <img src="styles/legend/disponibilidad_1_4.png" /> 14 - 39<br />'
        });

lyr_GoogleMaps_0.setVisible(true);lyr_disponibilidad_1.setVisible(true);
var layersList = [lyr_GoogleMaps_0,lyr_disponibilidad_1];
lyr_disponibilidad_1.set('fieldAliases', {'address': 'address', 'num': 'num', 'open': 'open', 'available': 'available', 'ticket': 'ticket', 'updated_at': 'updated_at', 'lat': 'lat', 'lon': 'lon', 'Num available bikes': 'Num available bikes', });
lyr_disponibilidad_1.set('fieldImages', {'address': 'TextEdit', 'num': 'Range', 'open': 'CheckBox', 'available': 'Range', 'ticket': 'CheckBox', 'updated_at': 'DateTime', 'lat': 'Hidden', 'lon': 'Hidden', 'Num available bikes': 'Hidden', });
lyr_disponibilidad_1.set('fieldLabels', {'address': 'inline label', 'num': 'inline label', 'open': 'inline label', 'available': 'inline label', 'ticket': 'inline label', 'updated_at': 'inline label', });
lyr_disponibilidad_1.on('precompose', function(evt) {
    evt.context.globalCompositeOperation = 'normal';
});