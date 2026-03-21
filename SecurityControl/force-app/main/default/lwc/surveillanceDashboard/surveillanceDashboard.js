import { LightningElement, track } from 'lwc';
import getLatestAlerts from '@salesforce/apex/SecurityDashboardController.getLatestAlerts';

export default class SurveillanceDashboard extends LightningElement {
    @track cameras = [
        { id: 'cam1', location: 'Main Entrance', statusText: 'Online', cardClass: 'cam-card', statusClass: 'status-label status-safe', hasThreat: false },
        { id: 'cam2', location: 'Parking Lot B', statusText: 'Online', cardClass: 'cam-card', statusClass: 'status-label status-safe', hasThreat: false },
        { id: 'cam3', location: 'Lobby Area', statusText: 'Online', cardClass: 'cam-card', statusClass: 'status-label status-safe', hasThreat: false },
        { id: 'cam4', location: 'Cafeteria', statusText: 'Online', cardClass: 'cam-card', statusClass: 'status-label status-safe', hasThreat: false },
        { id: 'cam5', location: 'Back Alley', statusText: 'Online', cardClass: 'cam-card', statusClass: 'status-label status-safe', hasThreat: false }
    ];

    timer;

    connectedCallback() {
        // Check for new alerts every 3 seconds
        this.timer = setInterval(() => {
            this.fetchAlerts();
        }, 3000);
    }

    disconnectedCallback() {
        clearInterval(this.timer);
    }

    fetchAlerts() {
        getLatestAlerts()
            .then(result => {
                this.processAlerts(result);
            })
            .catch(error => {
                console.error('Error fetching alerts', error);
            });
    }

    processAlerts(alerts) {
    // 1. Reset all cameras to 'Online' state
    const updatedCameras = this.cameras.map(cam => ({
        ...cam,
        statusText: 'Online',
        cardClass: 'cam-card',
        statusClass: 'status-label status-safe',
        hasThreat: false,
        videoUrl: null // Clear old video
    }));

    const seenCameras = new Set();

    // 2. Process alerts (Apex provides newest first)
    alerts.forEach(alert => {
        const camId = alert.Camera_ID__c;
        
        // Only process the NEWEST alert for each camera
        if (!seenCameras.has(camId)) {
            let camIndex = updatedCameras.findIndex(c => c.id === camId);
            if (camIndex !== -1) {
                updatedCameras[camIndex].statusText = '🚨 THREAT DETECTED';
                updatedCameras[camIndex].cardClass = 'threat-card';
                updatedCameras[camIndex].statusClass = 'status-label status-alert';
                updatedCameras[camIndex].hasThreat = true;
                updatedCameras[camIndex].message = alert.Message__c;
                updatedCameras[camIndex].confidence = alert.Confidence__c;
                updatedCameras[camIndex].weapon = alert.Weapon__c;
                
                // We will handle this URL in Phase 2
                updatedCameras[camIndex].videoUrl = alert.Video_ID__c ? 
                    `/sfc/servlet.shepherd/version/download/${alert.Video_ID__c}` : null;
            }
            seenCameras.add(camId);
        }
    });
    this.cameras = updatedCameras;
}
    }
