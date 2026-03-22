import { LightningElement, track } from 'lwc';
import getLatestAlerts from '@salesforce/apex/SecurityDashboardController.getLatestAlerts';

export default class SurveillanceCameras extends LightningElement {
   @track mapCenter;
    @track mapMarkers = [];
    @track selectedMarkerValue; // New variable to track selection
    
   @track cameras = [
    { 
        id: 'cam1', 
        location: 'Rockfort Temple Entrance', 
        lat: 10.8271, 
        lng: 78.6970, 
        statusText: 'Normal', 
        hasThreat: false, 
        iconName: 'utility:check', 
        iconVariant: 'success', 
        listClass: 'location-item state-safe',
        cardClass: 'cam-card',
        statusClass: 'status-label status-safe'
    },
    { 
        id: 'cam2', 
        location: 'Central Bus Stand', 
        lat: 10.7933, 
        lng: 78.6811, 
        statusText: 'Normal', 
        hasThreat: false, 
        iconName: 'utility:check', 
        iconVariant: 'success', 
        listClass: 'location-item state-safe',
        cardClass: 'cam-card',
        statusClass: 'status-label status-safe'
    },
    { 
        id: 'cam3', 
        location: 'Chathiram Bus Stand', 
        lat: 10.8296, 
        lng: 78.6917, 
        statusText: 'Normal', 
        hasThreat: false, 
        iconName: 'utility:check', 
        iconVariant: 'success', 
        listClass: 'location-item state-safe',
        cardClass: 'cam-card',
        statusClass: 'status-label status-safe'
    },
    { 
        id: 'cam4', 
        location: 'Srirangam Temple - North Tower', 
        lat: 10.8647, 
        lng: 78.6902, 
        statusText: 'Normal', 
        hasThreat: false, 
        iconName: 'utility:check', 
        iconVariant: 'success', 
        listClass: 'location-item state-safe',
        cardClass: 'cam-card',
        statusClass: 'status-label status-safe'
    },
    { 
        id: 'cam5', 
        location: 'Anna Nagar Main Road', 
        lat: 10.8030, 
        lng: 78.6900, 
        statusText: 'Normal', 
        hasThreat: false, 
        iconName: 'utility:check', 
        iconVariant: 'success', 
        listClass: 'location-item state-safe',
        cardClass: 'cam-card',
        statusClass: 'status-label status-safe'
    }
];

    connectedCallback() {
        this.initMarkers();
        setInterval(() => { this.fetchAlerts(); }, 3000);
    }

    initMarkers() {
        this.mapMarkers = this.cameras.map(cam => ({
            location: { Latitude: cam.lat, Longitude: cam.lng },
            title: cam.location,
            description: `Status: ${cam.statusText}`,
            value: cam.id // This 'value' must match cam.id
        }));
        
        // Initialize default center
        if (this.mapMarkers.length > 0 && !this.mapCenter) {
            this.selectedMarkerValue = this.mapMarkers[0].value;
            this.mapCenter = {
                location: { Latitude: this.cameras[0].lat, Longitude: this.cameras[0].lng }
            };
        }
    }

    handleLocationClick(event) {
        const camId = event.currentTarget.dataset.id;
        const selected = this.cameras.find(c => c.id === camId);
        
        if (selected) {
            // 1. Force the marker to be selected (opens the popup)
            this.selectedMarkerValue = camId;

            // 2. Create a NEW object reference to force map re-centering
            this.mapCenter = {
                location: { 
                    Latitude: selected.lat, 
                    Longitude: selected.lng 
                }
            };
            
            console.log('Centering map on:', selected.location);
        }
    }

    fetchAlerts() {
        getLatestAlerts().then(result => { this.processAlerts(result); });
    }

    processAlerts(alerts) {
        // ... (Keep your processAlerts logic from before)
        // Whenever this function finishes, call this.initMarkers() to update map descriptions
        // this.initMarkers();
    }
}