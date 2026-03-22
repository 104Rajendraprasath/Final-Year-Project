import { LightningElement, api, wire } from 'lwc';
import { getRecord, getFieldValue } from 'lightning/uiRecordApi';

// Import the Video ID field we created earlier
import VIDEO_ID_FIELD from '@salesforce/schema/Security_Alert__c.Video_ID__c';

export default class AlertVideoPlayer extends LightningElement {
    @api recordId; // Automatically gets the ID of the current record

    @wire(getRecord, { recordId: '$recordId', fields: [VIDEO_ID_FIELD] })
    securityAlert;

    get videoUrl() {
        const videoId = getFieldValue(this.securityAlert.data, VIDEO_ID_FIELD);
        if (videoId) {
            // This is the Salesforce URL to stream a file directly from ContentVersion
            return `/sfc/servlet.shepherd/version/download/${videoId}`;
        }
        return null;
    }
}