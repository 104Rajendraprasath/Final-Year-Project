public with sharing class SecurityDashboardController {
    @AuraEnabled(cacheable=false)
    public static List<Security_Alert__c> getLatestAlerts() {
        Datetime cutoff = Datetime.now().addMinutes(-10);
        return [
            SELECT Id, Camera_ID__c, Location__c, Confidence__c, Status__c, Threat_Type__c, Weapon__c ,Video_ID__c
            FROM Security_Alert__c
            WHERE CreatedDate >= :cutoff
            WITH SECURITY_ENFORCED
            ORDER BY CreatedDate DESC
        ];
    }
}
