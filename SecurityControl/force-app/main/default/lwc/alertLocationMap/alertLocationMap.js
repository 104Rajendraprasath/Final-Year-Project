import { LightningElement, api, wire } from 'lwc';
import { getRecord, getFieldValue } from 'lightning/uiRecordApi';

// Define the fields to fetch
const LAT_FIELD = 'Security_Alert__c.Latitude__c';
const LNG_FIELD = 'Security_Alert__c.Longitude__c';
const LOC_NAME = 'Security_Alert__c.Location__c';

export default class AlertLocationMap extends LightningElement {
    @api recordId;
    mapMarkers = [];

    @wire(getRecord, { recordId: '$recordId', fields: [LAT_FIELD, LNG_FIELD, LOC_NAME] })
    wiredRecord({ error, data }) {
        if (data) {
            const lat = getFieldValue(data, LAT_FIELD);
            const lng = getFieldValue(data, LNG_FIELD);
            const name = getFieldValue(data, LOC_NAME);

            this.mapMarkers = [{
                location: { Latitude: lat, Longitude: lng },
                title: name,
                description: `Incident reported at ${name}`
            }];
        }
    }
}