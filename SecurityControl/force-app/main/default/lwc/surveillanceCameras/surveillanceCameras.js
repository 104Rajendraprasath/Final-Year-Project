import { LightningElement, track } from 'lwc';
import getLatestAlerts from '@salesforce/apex/SecurityDashboardController.getLatestAlerts';

export default class SurveillanceCameras extends LightningElement {
   @track mapCenter;
    @track mapMarkers = [];
    @track selectedMarkerValue; // New variable to track selection
    
    @track cameras = [
        { id: 'cam1', location: 'Main Entrance', lat: 12.9716, lng: 77.5946, statusText: 'Normal', hasThreat: false, iconName: 'utility:check', iconVariant: 'success', listClass: 'location-item state-safe' },
        { id: 'cam2', location: 'Parking Lot B', lat: 13.0827, lng: 80.2707, statusText: 'Normal', hasThreat: false, iconName: 'utility:check', iconVariant: 'success', listClass: 'location-item state-safe' },
        { id: 'cam3', location: 'Lobby Area', lat: 19.0760, lng: 72.8777, statusText: 'Normal', hasThreat: false, iconName: 'utility:check', iconVariant: 'success', listClass: 'location-item state-safe' },
        { id: 'cam4', location: 'Cafeteria', lat: 28.6139, lng: 77.2090, statusText: 'Normal', hasThreat: false, iconName: 'utility:check', iconVariant: 'success', listClass: 'location-item state-safe' },
        { id: 'cam5', location: 'Back Alley', lat: 22.5726, lng: 88.3639, statusText: 'Normal', hasThreat: false, iconName: 'utility:check', iconVariant: 'success', listClass: 'location-item state-safe' }
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